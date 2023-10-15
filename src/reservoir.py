import torch
from .hebbian_learning import hebbian_wta

def generate_directed_ER(dim,
                         link_prob = 0.5,
                         weighted = True,
                         weight_range = (-1, 1),
                         spectral_radius = None):
    """
    Generate a directed Erdos-Renyi graph with a given link probability.
    
    Parameters
    ----------
    dim : int
        Number of nodes in the graph
    link_prob : float
        Probability of a link between two nodes
    weighted : bool
        Whether the graph is weighted or not
    weight_range : tuple
        Range of weights
    spectral_radius : float
        Spectral radius of the adjacency matrix
    
    Returns
    -------
    torch.Tensor
        Adjacency matrix
    """
    adj = torch.rand((dim, dim)) < link_prob
    if weighted:
        weights = torch.rand((dim, dim)) * (weight_range[1] - weight_range[0]) + weight_range[0]
        adj = adj.float() * weights

    # normalize to the given spectral radius
    if spectral_radius is not None:
        adj = adj / torch.max(torch.abs(torch.eig(adj)[0]))
        adj = adj * spectral_radius
    
    return adj.float()

class Reservoir(torch.nn.Module):
    """
    Reservoir computing layer. It has read-in and state weights, iniatilized in the
    class, but the read-out is an optional parameter.
    This is to offer flexibility in choice of readout (e.g. linear, MLP, k-NN, etc.),

    Parameters
    ----------
    in_dim : int
        Input dimensionality
    dim : int
        Reservoir dimensionality
    inertia : float
        Inertia of the reservoir update
    bias : bool
        Whether to include a bias term
    weight_range : tuple
        Range of initial weights
    bias_scale : tuple
        Range of initial bias
    spectral_radius : float
        Spectral radius of the adjacency matrix
    activation : torch.nn.Module
        Activation function
    state : torch.Tensor
        Initial reservoir state
    readout : torch.nn.Module
        Readout function
    """
    def __init__(self,
                 in_dim,
                 dim = None,
                 inertia = 0.5,
                 bias = True,
                 weight_range = (-1, 1),
                 bias_scale = (-1, 1),
                 spectral_radius = None,
                 activation = torch.nn.Tanh(),
                 state = None,
                 readout = None):
        super().__init__()
        self.in_dim = in_dim
        if dim is None:
            dim = in_dim
        self.dim = dim
        self.inertia = inertia

        self.in_weight = torch.nn.Parameter(torch.empty((dim, in_dim)))
        torch.nn.init.uniform_(self.in_weight, weight_range[0], weight_range[1])

        self.adj = torch.nn.Parameter(generate_directed_ER(dim,
                                                           spectral_radius = spectral_radius))
        self.adj.requires_grad = False

        # TODO : read literature on initialization of state
        if state is None:
            state = torch.randn(dim)
        self.state = torch.nn.Parameter(state)

        if bias:
            self.bias = torch.nn.Parameter(torch.empty(dim))
            torch.nn.init.uniform_(self.bias, bias_scale[0], bias_scale[1])
        else:
            self.bias = torch.zeros(dim)

        self.activation = activation

        self.readout = readout

    def forward(self, x, n_steps = 20, readout = True):
        """
        Parameters
        ----------
        x : torch.Tensor
            Input
        n_steps : int
            Number of reservoir update steps

        Returns
        -------
        torch.Tensor
            If readout is defined, return the readout of the reservoir state
            otherwise, return the state after n_steps
        """
        temp_state = self.state.clone().detach().repeat(x.shape[0], 1)
        for i in range(n_steps):
            new_state = self.activation(torch.nn.functional.linear(x, self.in_weight) + \
                                        torch.nn.functional.linear(self.state, self.adj) + \
                                        self.bias)
            temp_state.data.mul_(self.inertia).add_(new_state * (1 - self.inertia))

        # update state by averaging over the batch
        self.state.data = temp_state.data.clone().detach().mean(dim = 0)

        if readout and (self.readout is not None):
            return self.readout(temp_state)
        else:
            return temp_state.clone().detach()
        
    def train_step(self, x, labels, n_steps = 20):
        # TODO : what if the reservoir was Hebbian o_O
        hidden_state = self.forward(x,
                                    n_steps = n_steps,
                                    readout = False)
        
        error = self.readout.train_step(hidden_state,
                                        labels)
        return error
    
class LinearReadout(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 bias = False,
                 optimizer = None,
                 lr = 0.01
                 ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias = bias
        self.lr = lr

        self.W = torch.nn.Linear(in_dim, out_dim, bias = bias)

        if optimizer is not None:
            optimizer = optimizer(self.parameters(), lr = lr)
        self.optimizer = optimizer

    def forward(self, x):
        return self.W(x)
    
    def train_step(self, x, labels):
        labels = labels.float()
        if self.optimizer is None:
            with torch.no_grad():
                y = self.forward(x)
                loss = torch.nn.functional.cross_entropy(y, labels)

                # note WTA doesn't use the prediction in training
                dW = hebbian_wta(x, labels, self.W.weight)
                self.W.weight.data += self.lr * dW

                if self.bias:
                    self.W.bias.data -= self.lr * loss.mean(axis = 0)
                return [loss.item()]
        else:
            # enable grad
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()

                # only train the readout - let's be sure no gradients are propagated
                x = x.detach().clone().requires_grad_(True)
                y = self.forward(x)

                loss = torch.nn.functional.cross_entropy(y, labels)
                loss.backward()
                self.optimizer.step()
            return [loss.item()]
        
if __name__ == "__main__":
    from utils import mnist_test

    batch_size = 256
    in_dim = 784
    dim = 1024
    n_epochs = 3
    n_labels = 10
    save_every = 10
    bias = True
    activation = torch.nn.Tanh()
    optimizer = torch.optim.Adam

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    net = Reservoir(in_dim = in_dim,
                    dim = dim,
                    bias = bias,
                    activation = activation,
                    readout = LinearReadout(dim, 
                                            n_labels, 
                                            optimizer = optimizer)).to(device)

    accs, errors, y = mnist_test(net,
                                 device = device)