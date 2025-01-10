import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import make_interp_spline
from backprop_alts.reservoir import Reservoir, LinearReadout
from backprop_alts.predictive_coding import BDPredictiveCoder, PCNet
from backprop_alts.forwardforward import mnist_test_ff, PEPITA
from backprop_alts.utils import mnist_test

#TODO : plot layer details (like generalized eigenvalues) for each method
#TODO : plot of accuracy by depth for each method
#TODO : look into enabling bias for all models

class Baseline(torch.nn.Module):
    """
    The baseline model is a simple feedforward network, trained with backprop.

    To make it easily amenable with test utilitiy function, it has a trainstep method
    that does training.
    """
    def __init__(self,
                 dim,
                 out_dim,
                 dim_mult = 1,
                 n_layers = 3,
                 activation = torch.nn.ReLU(),
                 optimizer = torch.optim.SGD,
                 lr = 0.01,
                 ):
        super().__init__()
        self.in_dim = dim
        if isinstance(dim_mult, (int, float)):
            dim_mult = [dim_mult] * (n_layers - 1)
        self.dim_mult = dim_mult
        self.n_layers = n_layers

        self.layers = torch.nn.ModuleList()
        for i in range(n_layers - 1):
            dim_mult = self.dim_mult[i]
            self.layers.append(torch.nn.Linear(dim,
                                               int(dim * dim_mult),
                                               bias = False))
            dim = int(dim * dim_mult)

        self.layers.append(torch.nn.Linear(dim, out_dim, bias = False))
        
        self.activation = activation

        self.optimizer = optimizer(self.parameters(), lr = lr)

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
    
    def train_step(self, x_in, y):
        # remember grad is disabled by default
        with torch.set_grad_enabled(True):
            self.optimizer.zero_grad()
            x = x_in.clone().detach().requires_grad_(True)
            y_hat = self.forward(x)
            loss = torch.nn.functional.mse_loss(y_hat, y.float())
            loss.backward()
            self.optimizer.step()
        return [loss.item()]
    
def _calc_fair_reservoir_size(n_layers, dim_mult, in_dim, out_dim):
    """
    The reservoir is a massive blob of a network coupled to a readout network,
    so let's make sure its params are fair to other networks.
    """
    n_params = 0
    dim = in_dim
    if isinstance(dim_mult, (int, float)):
        dim_mult = [dim_mult] * (n_layers - 1)
    for i in range(n_layers - 1):
        new_dim = int(dim * dim_mult[i])
        n_params += dim * new_dim
        dim = new_dim
    n_params += dim * out_dim

    # n_params = reservoir_dim * (in_dim + out_dim) + reservoir_dim**2
    # quadratic formula: reservoir_dim = (-(in_dim + out_dim) + sqrt(in_dim + out_dim**2 + 4*n_params)) / 2
    reservoir_dim = (-(in_dim + out_dim) + np.sqrt((in_dim + out_dim)**2 + 4*n_params)) / 2
    return int(reservoir_dim)

def mnist_comparisons(n_epochs,
                      n_layers = 3,
                      in_dim = 784,
                      out_dim = 10,
                      mults = 0.75):
    
    if isinstance(n_layers, (int, float)):
        n_layers = [n_layers]
    if isinstance(mults, (int, float)):
        mults = [mults]
    assert len(n_layers) == len(mults), "n_layers and mults must be the same length"
    
    depth_scores = {}

    # TODO: i could probably do this in a loop or something
    for mult, layer_count in zip(mults, n_layers):
        scores = {}

        # test the baseline
        _, _, _, base = mnist_test(Baseline(in_dim, out_dim, 
                                            dim_mult = mult, 
                                            n_layers = layer_count),
                                   n_epochs = n_epochs)
        scores["Backprop"] = base

        #TODO : fix reservoir, need better input functionality for when in_dim > dim
        # test the reservoir
        res_dim = _calc_fair_reservoir_size(layer_count, mult, in_dim, out_dim)
        _, _, _, res = mnist_test(Reservoir(in_dim, res_dim, 
                                            adaptive = False,
                                            multi_ts = False,
                                            maintain_sparsity = False,
                                            # note the readout is allowed backprop
                                            readout= LinearReadout(res_dim, out_dim,
                                                                   optimizer=torch.optim.SGD,)),
                                  n_epochs = n_epochs)
        scores["Reservoir"] = res

        # test ff
        _, _, _, ff = mnist_test_ff(in_dim,
                                    layer_count,
                                    n_epochs,
                                    dim_mult = mult)
        scores["Forward-forward"] = ff

        # test pepita
        _, _, _, pep = mnist_test(PEPITA(in_dim,
                                         out_dim = out_dim,
                                         dim_mult = mult,
                                         n_layers = layer_count),
                                    n_epochs = n_epochs)
        scores["PEPITA"] = pep

        # test predictive coding
        _, _, _, pc = mnist_test(PCNet(in_dim, out_dim,
                                       dim_mult = mult,
                                       n_layers = layer_count,
                                       ipc = False),
                                 n_epochs = n_epochs)
        scores["Predictive Coding"] = pc

        # test adaptive relax predictive coding
        _, _, _, apc = mnist_test(PCNet(in_dim, out_dim,
                                        dim_mult = mult,
                                        n_layers = layer_count,
                                        ipc = False,
                                        adaptive_relaxation = True
                                        ),
                                  n_epochs = n_epochs)
        scores["Predictive Coding (Adap)"] = apc

        # test incremental predictive coding
        _, _, _, ipc = mnist_test(PCNet(in_dim, out_dim,
                                        dim_mult = mult,
                                        n_layers = layer_count,
                                        # slight variation needs lower LR, 0.1x
                                        lr = 0.001
                                        ),
                                  n_epochs = n_epochs)
        scores["Predictive Coding (Inc)"] = ipc

        # test bidirectional predictive coding
        _, _, _, bdp = mnist_test(BDPredictiveCoder(in_dim, out_dim,
                                                    dim_mult = mult,
                                                    n_layers = layer_count,),
                                  n_epochs = n_epochs)
        scores["Predictive Coding (BiDir)"] = bdp

        depth_scores[str(layer_count)] = scores
    return depth_scores

def pretty_plot(x_data,
                y_data,
                labels,
                plot_title,
                x_label,
                y_label,):
    os.makedirs("plots", exist_ok = True)
    fig, ax = plt.subplots(figsize = (7, 7))
    cmap = plt.get_cmap("rainbow")

    colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]
    for i, label in enumerate(labels):
        ax.scatter(x_data[i], y_data[i], 
                   color = colors[i], label = label,
                   marker = "o")
        x_smooth= np.linspace(np.min(x_data[i]), np.max(x_data[i]), 300) 

        # fitting curve
        spl = make_interp_spline(x_data[i], y_data[i],
                                 k = 2)
        y_smooth = spl(x_smooth)
        ax.plot(x_smooth, y_smooth, color = colors[i], alpha = 0.6)
    
    ax.legend(loc="lower right",)
    ax.set_facecolor("white")
    ax.grid(alpha = 0.1)
    ax.set_title(plot_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()

    plt.savefig(f"plots/{plot_title}.png")
    plt.close()

def get_mults(n_layers,
              in_dim,
              out_dim,
              total_neurons,
              merge_decay = 0.75):
    """
    Get the multipliers for each layer to hit the total number of neurons.
    """
    if isinstance(n_layers, (int, float)):
        n_layers = [n_layers]
    mults = []
    ratio = (total_neurons - out_dim) / in_dim
    for n_layers_i in n_layers:
        probs = np.array([merge_decay ** i for i in range(1, n_layers_i)])
        probs = probs / np.sum(probs)
 
        mult = list(ratio * probs)
        mults.append(mult)

    return mults

if __name__ == "__main__":
    n_epochs = 5
    n_layers = [4]
    in_dim = 784
    out_dim = 10
    # try to hit this number of neurons for comparability with different layer numbers
    total_neurons = 2000
    mults = get_mults(n_layers, in_dim, out_dim, total_neurons)

    all_scores = mnist_comparisons(n_epochs,
                                   mults = mults,
                                   n_layers = n_layers)

    for n_layers_i, scores in all_scores.items():
        # TODO : do this in pandas or something
        labels = list(scores.keys())
        accs = [list(scores[label]["epoch_accs"]) for label in labels]
        times = [list(scores[label]["epoch_times"]) for label in labels]
        times = np.cumsum(times, axis = 1)
        samples = [list(scores[label]["epoch_samples"]) for label in labels]
        samples = np.cumsum(samples, axis = 1)

        pretty_plot(times, accs, labels, f"Clock Time ({n_layers_i} layers)", "Time (s)", "Validation Accuracy")
        pretty_plot(samples, accs, labels, f"Sample Efficiency({n_layers_i} layers)", "Samples", "Validation Accuracy")
        

