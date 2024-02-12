import torch
from hebbian_learning import hebbian_pca

class BDPredictiveBlock(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 activation = torch.nn.Tanh(),
                 bias = False,
                 whiten = False):
        """
        A bidirectional predictive coding block. Has a forward and backward layer.
        Weights update with the anti-Hebbian learning rule.

        Parameters
        ----------
        in_dim : int
            The input dimension of the block.
        out_dim : int
            The output dimension of the block.
        activation : torch.nn.Module
            The activation function to use for each layer.
        bias : bool
            Whether to use a bias term for each layer.
        whiten : bool
            Whether to use a whitening term (i.e. variance normalizing) for each layer.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.forward_layer = torch.nn.Linear(in_dim, out_dim, bias = False)
        self.backward_layer = torch.nn.Linear(out_dim, in_dim, bias = False)

        self.sd_forward = torch.nn.Parameter(torch.ones(in_dim))
        self.sd_backward = torch.nn.Parameter(torch.ones(out_dim))
        self.whiten = whiten

        self.bias_forward = torch.nn.Parameter(torch.zeros(in_dim))
        self.bias_backward = torch.nn.Parameter(torch.zeros(out_dim))
        self.biased = bias
        
        self.activation = activation

    def forward(self, x):
        output = self.forward_layer((x + self.bias_forward) / self.sd_forward)
        output = self.activation(output)
        return output
    
    def backward_step(self, x, noise_scale = 0):
        output = self.backward_layer((x + self.bias_backward) / self.sd_backward)
        output = self.activation(output)
        if noise_scale > 0:
            output += torch.randn_like(output) * noise_scale
        return output
    
    def update_forward(self, error, x, lr = 0.01):
        batch_size = x.shape[0]
        dW = torch.einsum("bi,bj->ji", x, error) / batch_size
        
        self.forward_layer.weight -= lr * dW

        if self.whiten:
            self.sd_forward .mul_(1 - lr).add_(lr * x.std(dim = 0))

        if self.biased:
            db = -(x.mean(dim = 0))
            self.bias_forward.mul_(1 - lr).add_(lr * db)

    def update_backward(self, error, x, lr = 0.01):
        batch_size = x.shape[0]
        dW = torch.einsum("bi,bj->ij", error, x) / batch_size

        self.backward_layer.weight -= lr * dW

        if self.whiten:
            self.sd_backward.mul_(1 - lr).add_(lr * x.std(dim = 0))

        if self.biased:
            db = -(x.mean(dim = 0))
            self.bias_backward.mul_(1 - lr).add_(lr * db)
        
class BDPredictiveCoder(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 dim_mult = 1,
                 n_layers = 3,
                 activation = torch.nn.Tanh(),
                 bias = False):
        """
        A bidirectional predictive coding network with forward and backward layers that 
        learns via Hebbian learning. The forward layers pass predictions forward and the
        backward layers pass errors back. Both learn with the generalized Hebbian
        learning rule, so that any odd activation function can be used.

        Parameters
        ----------
        in_dim : int
            The input dimension of the network.
        out_dim : int
            The output dimension of the network.
        dim_mult : float or list of floats
            The dimension multiplier for each layer. If a float, the same value
            is used for each layer. If a list, the length must be n_layers - 1
        n_layers : int
            The number of layers in the network.
        activation : torch.nn.Module
            The activation function to use for each layer.
        bias : bool
            Whether to use a bias term for each layer.
        """
        super().__init__()
        self.in_dim = in_dim
        if isinstance(dim_mult, (int, float)):
            dim_mult = [dim_mult] * (n_layers - 1)
        self.dim_mult = dim_mult
        self.n_layers = n_layers

        self.layers = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            dim_mult = self.dim_mult[i]
            self.layers.append(BDPredictiveBlock(in_dim, 
                                                      int(in_dim * dim_mult),
                                                      activation = activation,
                                                      bias = bias))
            in_dim = int(in_dim * dim_mult)
        self.layers.append(BDPredictiveBlock(in_dim, 
                                                  out_dim,
                                                  activation = activation,
                                                  bias = bias))
        

        self.requires_grad_(False)

    def forward(self, x, return_all = False):
        if return_all:
            outputs = [x]
            for layer in self.layers:
                x = layer(x)
                outputs.append(x)
            return outputs
        else:
            for layer in self.layers:
                x = layer(x)
            return x
    
    def backward_step(self, x, return_all = False, noise_scale = 0):
        if return_all:
            outputs = []
            for layer in reversed(self.layers):
                x = layer.backward_step(x)
                outputs.append(x)
            return outputs
        else:
            for i, layer in enumerate(reversed(self.layers)):
                if isinstance(noise_scale, (list, tuple)):
                    noise_scale_i = noise_scale[i]
                else:
                    noise_scale_i = noise_scale
                x = layer.backward_step(x, noise_scale = noise_scale_i)
            return x
    
    def train_step(self, x, y, 
                   n_iters = 20, 
                   lr_per_step = 0.01,
                   noise_scale = 0.1):
        """
        This method uses a train step based on the book Gradient Expectations.
        It uses no derivatives.
        """
        x_clamp = x.clone().detach()
        y_clamp = y.clone().detach()

        lr = lr_per_step / n_iters

        for i in range(n_iters):
            x_hats = self.forward(x_clamp, return_all = True)
            error = x_hats[-1] - y_clamp
            for j in range(self.n_layers, 0, -1):
                layer = self.layers[j - 1]
                x_hat_prev = x_hats[j - 1]
                x_hat = x_hats[j]

                layer.update_forward(error, x_hat_prev, lr = lr)
                error = layer.backward_step(error,
                                            noise_scale = noise_scale) + x_hat_prev
                layer.update_backward(error, x_hat, lr = lr)

        return error
    
class BDLBlock(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 dim,
                 back_dim,
                 activation = torch.nn.Tanh(),
                 bias = False,
                 whiten = False,
                 stateful = True):
        """
        A bidirectional learning block. Has forward, backward, and lateral layers.

        Parameters
        ----------
        in_dim : int
            The input dimension of the block.
        dim : int
            The output dimension of the block.
        back_dim : int
            The dimension of the backward input.
        activation : torch.nn.Module
            The activation function to use for each layer.
        bias : bool
            Whether to use a bias term for each layer.
        whiten : bool
            Whether to use a whitening term (i.e. variance normalizing) for each layer.
        """
        super().__init__()

        self.in_dim = in_dim
        self.dim = dim
        self.back_dim = back_dim
        self.dim = dim

        self.forward_layer = torch.nn.Linear(in_dim, dim, bias = False)
        self.backward_layer = torch.nn.Linear(back_dim, dim, bias = False)
        self.lateral_layer = torch.nn.Linear(dim, dim, bias = False)

        self.sd_forward = torch.nn.Parameter(torch.ones(in_dim))
        self.sd_backward = torch.nn.Parameter(torch.ones(back_dim))
        self.whiten = whiten

        self.bias_forward = torch.nn.Parameter(torch.zeros(in_dim))
        self.bias_backward = torch.nn.Parameter(torch.zeros(back_dim))
        self.biased = bias

        self.stateful = stateful
        self.register_buffer("state", torch.zeros(dim))
        
        self.activation = activation

    def forward(self, x, y, lateral = None, return_all = False):
        """
        Parameters
        ----------
        x : torch.Tensor
            The forward input.
        y : torch.Tensor
            The backward input.
        lateral : torch.Tensor
            The lateral input.
        return_all : bool
            Whether to hidden activation as well as the output.
        """
        x_h = self.forward_layer((x + self.bias_forward) / self.sd_forward)
        if y is None:
            y = torch.zeros(self.back_dim, device = x.device)
        y_h = self.backward_layer((y + self.bias_backward) / self.sd_backward)
        if lateral is None:
            lateral_h = self.lateral_layer(self.state)
        else:
            lateral_h = self.lateral_layer(lateral)

        h = x_h - y_h - lateral_h
        output = self.activation(h)
        if self.stateful:
            self.state = output

        if return_all:
            return output, h
        return output
    
    def train_step(self, x, y, lateral = None, lr = 0.01):
        """
        Update weights based on a predictive coding energy functional that encourages
        lateral and backward predictions to match the forward predictions.
        """
        if lateral is None:
            lateral = self.state
        output, h = self.forward(x, y,
                                 lateral = lateral, return_all = True)

        dF = hebbian_pca(x, h, self.forward_layer.weight)
        dB = -hebbian_pca(y, h, self.backward_layer.weight)
        dL = -hebbian_pca(lateral, h, self.lateral_layer.weight)

        self.forward_layer.weight += lr * dF
        self.backward_layer.weight += lr * dB
        self.lateral_layer.weight += lr * dL
        
        return output
    
class BDLNet(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 dim_mult = 1,
                 n_layers = 3,
                 activation = torch.nn.Tanh(),
                 bias = False):
        """
        A bidirectional learning network with forward, backward, and lateral layers that 
        learns via Hebbian learning. The forward layers pass predictions forward and the
        backward layers pass errors back.
        """
        super().__init__()
        self.layers = torch.nn.ModuleList()

        dims = [in_dim] + [int(in_dim * (dim_mult ** i)) for i in range(n_layers - 1)] + [out_dim]

        for i in range(n_layers - 1):
            self.layers.append(BDLBlock(dims[i],
                                        dims[i + 1],
                                        dims[i + 2],
                                        activation = activation,
                                        bias = bias,
                                        # require stateful for full net
                                        stateful = True))
            
    def reduce_state(self):
        """
        When state is a batch, convert to a vector.
        """
        for layer in self.layers:
            if len(layer.state.shape) > 1:
                layer.state = layer.state.mean(dim = 0)

        
    def forward(self,
                x,
                target = None,
                reduce_state = True,
                n_iters = 3,):
        for i in range(n_iters):
            x_i = x.clone().detach()
            for j in range(len(self.layers) - 1):
                y = self.layers[j + 1].state
                x_i = self.layers[j](x_i, y, return_all = False)

            out = self.layers[-1](x_i, target, return_all = False)

        if reduce_state:
            self.reduce_state()

        return out
    
    def train_step(self, x, target, lr = 0.01):
        prediction = self.forward(x,
                                  target = target,
                                  reduce_state = False,)
        states = [layer.state for layer in self.layers]
        states = [x] + states + [-target.float()]

        for i in range(len(self.layers)):
            layer = self.layers[i]
            x_i = states[i]
            y = states[i + 2]

            output = layer.train_step(x_i, y, lr = lr)

        self.reduce_state()
        return output


class PCNet(torch.nn.Module):
    """
    A predictive coding network, as described here:
    https://www.biorxiv.org/content/10.1101/2022.05.17.492325v2

    It is a feedforward network with linear layers. Learns via predictive coding
    with a target relaxation step, which is where the backward pass comes in. The 
    weight updates are Hebbian. For this reason, any activation functions must be
    odd.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 dim_mult = 1,
                 n_layers = 3,
                 activation = torch.nn.Tanh(),
                 adaptive_relaxation = False
                 ):
        super().__init__()
        self.in_dim = in_dim
        if isinstance(dim_mult, (int, float)):
            dim_mult = [dim_mult] * (n_layers - 1)
        self.dim_mult = dim_mult
        self.n_layers = n_layers
        self.adaptive_relaxation = adaptive_relaxation

        self.layers = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            dim_mult = self.dim_mult[i]
            self.layers.append(torch.nn.Linear(in_dim,
                                               int(in_dim * dim_mult),
                                               bias = False))
            in_dim = int(in_dim * dim_mult)

        self.layers.append(torch.nn.Linear(in_dim, out_dim, bias = False))
        
        self.activation, self.activation_derivative = self._init_activation(activation)

        self.requires_grad_(False)

    def forward(self, x):
        """Forward pass through the network, returns the final prediction.
        Note that in the PCN framework, activations are applied before each
        layer, so we apply the activation function here."""
        for layer in self.layers:
            x = layer(self.activation(x))
        return x
    
    def _init_activation(self, activation):
        """Creates derivatives for the various activation functions. Using
        autograd or jacobians seemed much slower, so specifying them manually.
        """
        if isinstance(activation, torch.nn.Identity):
            activation_derivative = lambda x: torch.ones_like(x)
        elif isinstance(activation, torch.nn.Tanh):
            tanh = torch.nn.functional.tanh
            activation_derivative = lambda x: 1 - tanh(x) ** 2
        else:
            raise NotImplementedError(f"Activation {activation} not implemented")
        return activation, activation_derivative
        
    def train_step(self, x, y, 
                   n_iters = 128, 
                   lr = 0.01,
                   equilibration_lr = 0.1,
                   noise_scale = 0.1):
        """
        Predictive coding training step, see Algorithm 1 in the paper:
        https://www.biorxiv.org/content/10.1101/2022.05.17.492325v2
        """
        #TODO : all variables can be stored as tensor buffers
        x_clamp = x.clone().detach()
        y_clamp = y.clone().detach()

        activations = [x_clamp]
        activations += [torch.zeros(x_clamp.shape[0], 
                                    layer.weight.shape[0], 
                                    device = x_clamp.device) for layer in self.layers]
        activations[-1] = y_clamp

        backward_weights = [layer.weight.T for layer in self.layers]

        # equilibration stage
        for i in range(n_iters):
            # this is different from paper, i merged the forward + update loops to 1 for efficiency
            errors = []
            for j in range(self.n_layers):
                layer = self.layers[j]
                estimate = layer(self.activation(activations[j]))
                error = activations[j + 1] - estimate
                errors.append(error)
                
                if j == 0:
                    continue
                
                back_weight = backward_weights[j]
                grad = self.activation_derivative(activations[j])
                dx = grad * (back_weight @ errors[j].T).T

                if self.adaptive_relaxation:
                    back_weight += lr * equilibration_lr * (activations[j].T @ errors[j]) / x.shape[0]
                    backward_weights[j] = back_weight

                activations[j] += equilibration_lr * (dx - errors[j - 1])

        # update weights after equilibration
        for i in range(self.n_layers):
            dW = lr * torch.einsum("bi,bj->ij", errors[i], activations[i]) / x.shape[0]
            self.layers[i].weight += dW

        return error # final error

#TODO : the backwards prediction is always the same for each digit for BDPredictiveCoder?
#TODO : can i add intelligent stochasticity for variable prediction?
#TODO : eg learn a noise direction based on the current input?
#TODO : variable LR based on surprisal?
#TODO : adding more layers reduces the accuracy?
#TODO : add GeneRec
if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    import numpy as np
    from utils import mnist_test

    batch_size = 256
    in_dim = 784
    dim_multiplier = 0.33
    n_epochs = 3
    n_labels = 10
    sample_noise_scale = 0.01
    bias = True
    whiten = True
    n_layers = 3
    activation = torch.nn.Tanh()
    adaptive_relax = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # net = PCNet(in_dim,
    #             n_labels,
    #             dim_multiplier,
    #             activation = activation,
    #             n_layers = n_layers,
    #             adaptive_relaxation = adaptive_relax).to(device)
    net = BDLNet(in_dim,
                  n_labels,
                  dim_multiplier,
                  n_layers = n_layers).to(device)
    accs, errors, y, details = mnist_test(net,
                                device = device)

    # plots the backward predictions
    if isinstance(net, BDPredictiveCoder):
        y_float = y.float()
        gen_noise = torch.randn_like(y_float) * sample_noise_scale
        gens = net.backward_step(-y_float + gen_noise)
        # undo normalization
        gens = gens * 0.5 + 0.5
        gens = gens.view(-1, 1, 28, 28)
        plt.figure(figsize = (10, 10))
        plt.imshow(make_grid(gens).permute(1, 2, 0).cpu().detach().numpy())

        plt.figure(figsize = (10, 10))
        errors = torch.stack(errors, dim = 0)
        errors = errors * 0.5 + 0.5
        errors = errors.view(-1, 1, 28, 28)

        plt.imshow(make_grid(errors).permute(1, 2, 0).cpu().detach().numpy())







