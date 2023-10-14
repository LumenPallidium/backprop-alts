import torch
import numpy as np

def goodness(activation, theta = 0):
    energy = torch.sum(activation ** 2) - theta
    inverse_goodness = (1 + torch.exp(-energy))
    return 1 / inverse_goodness

class FFBlock(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 activation = torch.nn.ReLU(),
                 theta = 0,
                 bias = False):
        super().__init__()

        self.theta = theta
        self.activation = activation

        self.W = torch.nn.Linear(in_dim, out_dim, bias = bias)

    def forward(self, x):
        return self.activation(self.W(x))
    
    def forward_goodness(self, weight, x):
        x = torch.nn.functional.linear(x, weight)
        x = self.activation(x)
        return goodness(x, theta = self.theta)
    
    def train_step(self, x, labels, lr = 0.01):
        goodness_grad = torch.func.jacrev(self.forward_goodness)
        dW = torch.func.vmap(goodness_grad, 
                             in_dims = (None, 0))(self.W.weight, x)
        # maximize goodness for positive labels, min for negative (by multiplying by label)
        dW = torch.mean(dW * labels[:, None, None], axis = 0)
        self.W.weight.data += lr * dW

class FFNet(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 n_layers = 3,
                 dim = None,
                 out_dim = None,
                 activation = torch.nn.ReLU(),
                 theta = 0,
                 bias = False,
                 n_labels = 10):
        super().__init__()
        self.in_dim = in_dim
        self.n_layers = n_layers
        if dim is None:
            dim = in_dim
        self.dim = dim
        if out_dim is None:
            out_dim = dim
        self.out_dim = out_dim
        self.activation = activation
        self.theta = theta
        self.n_labels = n_labels

        self.layers = torch.nn.ModuleList()
        self.layers.append(FFBlock(in_dim + n_labels, dim, activation = activation, theta = theta, bias = bias))

        for i in range(n_layers - 2):
            self.layers.append(FFBlock(dim, dim, activation = activation, theta = theta, bias = bias))
        
        self.layers.append(FFBlock(dim, out_dim, activation = activation, theta = theta, bias = bias))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def train_step(self, x, y, layer_n, lr = 0.01):
        y = torch.nn.functional.one_hot(y, num_classes = self.n_labels)

        # create an array like y, but not the same label
        y_neg = torch.rand(y.shape, device=y.device) * (1.0 - y)
        y_neg = torch.nn.functional.one_hot(torch.argmax(y_neg, axis=1), num_classes = n_labels)

        # create a random mask
        mask = torch.rand(y.shape[0], device=x.device) > 0.5

        y[mask] = y_neg[mask]
        # vector that is +1 for pos, -1 for neg samples
        pos_neg_lab = (mask.float() * -2) + 1

        x = torch.cat([x, y], axis = -1)

        for i in range(layer_n):
            x = self.layers[i](x)
        
        self.layers[layer_n].train_step(x, pos_neg_lab, lr = lr)

    def validate(self, x, y):
        """
        To validate, we must look at energies for all possible labels.
        
        Parameters
        ----------
        x : torch.Tensor
            Input data, batched
        y : torch.Tensor
            Labels
        """
        # generate one-hots for all possible labels
        labels = torch.arange(self.n_labels, device = x.device)
        labels = torch.nn.functional.one_hot(labels, num_classes = self.n_labels)
        labels = labels.unsqueeze(1).repeat([1, x.shape[0], 1])

        # repeat input for each label and concat label to data
        x = x.repeat([self.n_labels, 1, 1])
        x = torch.cat([x, labels], axis = -1)

        # prediction is mean energy for each label
        pred = self.forward(x).mean(dim = -1)
        pred = pred.argmax(dim = 0)

        return (pred == y).float().mean().item()
    
if __name__ == "__main__":
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from torchvision import datasets, transforms
    from utils import _prepare_for_epochs

    in_dim = 784
    dim = 512
    threshold = 10
    n_epochs = 3
    n_labels = 10
    batch_size = 256
    n_layers = 3
    lr = 0.1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FFNet(in_dim,
                dim = dim,
                n_layers = n_layers,
                theta = threshold,
                ).to(device)

    with torch.no_grad():
        mnist, mnist_val, accs, errors = _prepare_for_epochs()

        for epoch in range(n_epochs):

            epoch_val_accs = []

            val_loader = torch.utils.data.DataLoader(mnist_val, 
                                                    batch_size = batch_size, 
                                                    shuffle = True)
            for i, (x, y) in tqdm(enumerate(val_loader)):
                x = x.reshape([x.shape[0], -1]).to(device)
                y = y.to(device)

                acc = net.validate(x, y)
                epoch_val_accs.append(acc)

            print(f"Epoch {epoch} validation accuracy: {np.mean(epoch_val_accs)}")
            accs.extend(epoch_val_accs)

            for layer_i in range(n_layers):

                train_loader = torch.utils.data.DataLoader(mnist, 
                                                           batch_size = batch_size, 
                                                           shuffle = True)
                for i, (x, y) in tqdm(enumerate(train_loader)):
                    x = x.reshape([x.shape[0], -1]).to(device)
                    y = y.to(device)

                    net.train_step(x, y, layer_i, lr = lr)



