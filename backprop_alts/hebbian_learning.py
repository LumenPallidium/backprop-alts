import torch
import einops
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def hebbian_wta(x, y, W):
    """
    Function that computes Hebbian weight update in the winner-take-all
    formulation (soft). Note that in n-dimensional space, the winner-take-all
    tends to find <n clusters, with a tendency to get stuck in-between clusters
    when the number of clusters is greater than n.
    
    This function does use the FastHebb formulation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, input_dim)
    y : torch.Tensor
        Output tensor of shape (batch_size, output_dim)
    W : torch.Tensor
        Weight tensor of shape (input_dim, output_dim)
    """
    y = torch.functional.F.softmax(y, dim = -1)

    c = y / y.sum(dim = 0, keepdim = True)
    cr = (c * y)

    pre_post_correlation = torch.einsum("bi,bj->ij", cr, x)

    weight_expectation = cr.sum(dim = 0).unsqueeze(-1) * W

    weight_update = pre_post_correlation - weight_expectation
    return weight_update

def hebbian_pca(x, y, W):
    """
    Function that computes Hebbian weight update in the PCA formulation. 
    Note y can have been passed through a nonlinearity, so long as it is
    an odd function.

    This function does use the FastHebb formulation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, input_dim)
    y : torch.Tensor
        Output tensor of shape (batch_size, output_dim)
    W : torch.Tensor
        Weight tensor of shape (input_dim, output_dim)
    """
    batch_size = x.shape[0]
    pre_post_correlation = torch.einsum("bi,bj->ij", y, x)
    post_product = torch.einsum("bi,bj->ij", y, y).tril()

    weight_expectation = torch.einsum("op,oi->pi", post_product, W)

    weight_update = (pre_post_correlation - weight_expectation) / batch_size
    return weight_update

class HebbianConv2d(torch.nn.Module):
    """
    Hebbian convolutional layer. This layer is implemented as a torch.nn.Conv2d
    layer with a custom weight update function. The weight update function is
    the FastHebb formulation, which is a more numerically stable version of
    the Hebbian learning rule.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    kernel_size : int
        Size of the convolutional kernel
    stride : int
        Stride of the convolutional kernel
    padding : int
        Padding of the convolutional kernel
    dilation : int
        Dilation of the convolutional kernel
    bias : bool
        If True, then a predictive coding bias is added i.e
        a bias so that output has zero mean and unit variance
        (per channel)
    bias_alpha : float
        Update rate for the predictive coding bias
    activation : torch.nn.Module
        Activation function to apply to the output
    wta : bool
        If True, then a winner-take-all rule (softhebb is applied)
        https://iopscience.iop.org/article/10.1088/2634-4386/aca710/pdf
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0,
                 bias = True,
                 bias_alpha = 0.5,
                 activation = None,
                 wta = False):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.wta = wta
        if wta:
            print("Warning: WTA not compatible with negative activation functions")

        if activation is None:
            self.activation = torch.nn.Identity()
        else:
            self.activation = activation

        weight_scale = (1 / np.sqrt(in_channels * kernel_size[0] * kernel_size[1]))
        weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1]) * weight_scale
        self.weight = torch.nn.Parameter(weight)

        self.register_buffer("bias", torch.zeros(1, in_channels, 1, 1))
        self.register_buffer("sd", torch.ones(1, in_channels, 1, 1))
        self.bias_alpha = bias_alpha
        if bias:
            self.update_bias = True
        else:
            self.update_bias = False

    def forward(self, x):
        x = (x - self.bias) / (self.sd + 1e-3)
        y = torch.nn.functional.conv2d(x, self.weight, stride = self.stride, padding = self.padding)
        return self.activation(y)
    
    def _update_bias(self, x):
        alpha = 1 - self.bias_alpha
        # try and make the bias equal the mean
        self.bias.mul_(alpha).add_(self.bias_alpha * x.mean(dim = (0, 2, 3), keepdim = True))
        # if current SD is too low, then increase etc
        self.sd.mul_(alpha).add_(self.bias_alpha * x.std(dim = (0, 2, 3), keepdim = True))
        
    def train_step(self, x, lr = 0.01):
        x_raw = x.clone()
        x = (x - self.bias) / (self.sd + 1e-3)
        batch_size = x.shape[0]
        x_unfolded = torch.nn.functional.unfold(x, 
                                                kernel_size = self.kernel_size, 
                                                stride = self.stride, 
                                                padding = self.padding)
        n_blocks = x_unfolded.shape[-1]
        x_unfolded = einops.rearrange(x_unfolded, 
                                      "...(c k1 k2) n -> ... c n k1 k2", 
                                      c = self.in_channels, 
                                      k1 = self.kernel_size[0], 
                                      k2 = self.kernel_size[1])
                                      
        y =  torch.nn.functional.conv2d(x, self.weight, stride = self.stride, padding = self.padding)
        y = self.activation(y)
        y_unfolded = einops.rearrange(y, "... c h w -> ... c (h w)")

        if self.wta:
            y_unfolded = torch.functional.F.softmax(y_unfolded,
                                                    dim = -1)

            c = y_unfolded / y_unfolded.sum(dim = 0, keepdim = True)
            y_unfolded = (c * y_unfolded)

        # sum over batch and image chunks
        dW = torch.einsum("bcnij, bkn -> kcij", x_unfolded, y_unfolded)
        # get weight expectation
        y_outer = torch.einsum("bin, bjn -> ij", y_unfolded, y_unfolded)
        weight_expectation = torch.einsum("ac,ckij->akij", y_outer, self.weight) / self.out_channels
        dW -= weight_expectation 
        dW /= (batch_size * n_blocks)

        if (torch.isnan(dW).any()):
            raise ValueError("NaN encountered in weight update")

        self.weight += lr * dW
        if self.update_bias:
            self._update_bias(x_raw)
        return y
    
class HebbEncoder(torch.nn.Module):
    def __init__(self,
                 kernel_sizes = [3, 5, 7],
                 strides = [2, 3, 4],
                 channel_sizes = [1, 32, 64, 128],
                 linear_dim = 512,
                 out_dim = 10,
                 activation = torch.nn.GELU()):
        super().__init__()
        conv = []
        for i, kernel_size in enumerate(kernel_sizes):
            padding = (kernel_size - 1) // 2
            conv.append(HebbianConv2d(channel_sizes[i], 
                                      channel_sizes[i+1],
                                      kernel_size,
                                      stride = strides[i],
                                      padding = padding,
                                      activation = activation))
        self.conv = torch.nn.Sequential(*conv)
        linear = torch.randn(linear_dim, out_dim)
        self.linear = torch.nn.Parameter(linear)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = x @ self.linear
        return x
    
    def train_step(self, x, expected_output = None, lr = 0.01):
        y = x.clone()
        for layer in self.conv:
            y = layer.train_step(y, lr = lr)
        y = y.reshape(x.shape[0], -1)

        if expected_output is not None:
            dW = hebbian_pca(y, expected_output, self.linear.T)
            self.linear += lr * dW.T
        else:
            out = y @ self.linear
            dW = hebbian_pca(y, out, self.linear.T)
            self.linear += lr * dW.T
        return x

def simple_test_plots(weights, pca = False, scale_max = 5, n_clusters = 3, centers = None, points = None):
    # plot weights and centers
    fig, ax = plt.subplots(figsize = (7, 7))

    if pca:
        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, 2)]
        # doing this in np for convenience
        weight_list = np.stack(weights).tolist()

        alpha = 0

        for weight in weight_list:
            alpha += 0.5 / (len(weight_list) + 1)
            ax.plot([0, weight[0][0] * scale_max], 
                    [0, weight[0][1] * scale_max], 
                    color = colors[0], alpha = alpha)
            ax.plot([0, weight[1][0] * scale_max], 
                    [0, weight[1][1] * scale_max], 
                    color = colors[1], alpha = alpha)

        points = np.stack(points).transpose().tolist()
        ax.scatter(points[0], points[1])

        ax.set_xlim(-2 * scale_max, 2 * scale_max)
        ax.set_ylim(-2 * scale_max, 2 * scale_max)

    else:
        cmap = plt.get_cmap("rainbow")
        colors = [cmap(i) for i in np.linspace(0, 1, n_clusters)]

        # doing this in np for convenience
        weight_list = np.stack(weights).transpose(1, 2, 0).tolist()

        for weight, color in zip(weight_list, colors):
            ax.plot(weight[0], weight[1], color = color)

        ax.scatter(centers[:, 0].cpu().numpy(), centers[:, 1].cpu().numpy(), marker = "x", label = "centers")
    ax.legend()

def test_simple(n_iters = 1000,
                lr = 0.1,
                scale_max = 5,
                space_dim = 2,
                n_clusters = 3,
                batch_size = 256,
                pca = False):
    """
    Function that tests the Hebbian learning rule on a simple problem.
    If pca is True, then the problem is PCA, otherwise it is WTA (clustering).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if pca:
        n_clusters = space_dim

    with torch.no_grad():
        # test involving finding five clusters
        lin = torch.nn.Linear(space_dim, n_clusters, bias = False).to(device)
        intl_weight = lin.weight.clone()

        centers = torch.randn(n_clusters, space_dim, device = device) * scale_max
        centers = centers - centers.mean(dim = 0, keepdim = True)
        # to scale and rotate for PCA tests
        scales = torch.randint(1, scale_max, size = (space_dim,), device = device).float()
        q, _ = torch.linalg.qr(torch.randn(space_dim, space_dim, device = device))

        weights = []
        points = []
        for i in tqdm(range(n_iters)):

            if pca:
                x = torch.randn(batch_size, space_dim, device = device)
                x = x * (q @ scales).unsqueeze(0)

            else:
                x = torch.randn(batch_size, n_clusters, space_dim, device = device)
                x = x + centers.unsqueeze(0)

                indx = torch.randint(0, n_clusters, (batch_size, ), device = device)
                x = x[torch.arange(batch_size), indx]

            y = lin(x)

            if pca:
                dW = hebbian_pca(x, y, lin.weight)
            else:
                dW = hebbian_wta(x, y, lin.weight)

            lin.weight += lr * dW

            points.append(x[0].clone().detach().cpu().numpy())
            weights.append(lin.weight.clone().detach().cpu().numpy())

        print(f"Weight change after {n_iters} iterations: {torch.norm(lin.weight - intl_weight)}")

        simple_test_plots(weights, pca = pca, scale_max = scale_max, n_clusters = n_clusters, centers = centers, points = points)

#TODO : clean this up, split up
def test_conv(n_epochs = 10,
              kernel_sizes = [3, 5, 7],
              strides = [2, 3, 4],
              linear_size = 512,
              batch_size = 256,
              channel_sizes = [1, 32, 64, 128],
              channels_to_plot = 32,
              lr = 0.001,
              plot = True):
    """
    Function that tests the Hebbian convolution, by plotting the filters.
    """
    from utils import _prepare_for_epochs
    from torchvision.utils import make_grid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        net = HebbEncoder(kernel_sizes = kernel_sizes,
                          strides = strides,
                          channel_sizes = channel_sizes,
                          linear_dim = linear_size,
                          out_dim = 10).to(device)
        initial_weights = [x.weight.clone().detach().cpu() for x in net.conv]
        mnist, mnist_val, accs, errors = _prepare_for_epochs()

        for epoch in range(n_epochs):
            train_loader = torch.utils.data.DataLoader(mnist, 
                                                        batch_size = batch_size, 
                                                        shuffle = True)
            for i, (x, label) in tqdm(enumerate(train_loader)):
                step_batch = x.shape[0]
                x, label = x.to(device), label.to(device)
                label_onehot = torch.nn.functional.one_hot(label, 10).float()
                x = x.view(-1, 1, 28, 28)
                net.train_step(x,
                               expected_output = label_onehot,
                               lr = lr)

            accs = []
            val_loader = torch.utils.data.DataLoader(mnist_val,
                                                     batch_size = batch_size,
                                                     shuffle = True)
            for i, (x, label) in tqdm(enumerate(val_loader)):
                x, label = x.to(device), label.to(device)
                x = x.view(-1, 1, 28, 28)

                label_hat = net(x)
                label_hat = torch.argmax(label_hat, dim = -1)
                acc = (label == label_hat).float().mean().item()
                accs.append(acc)
            print(f"Epoch {epoch + 1} / {n_epochs} - Accuracy: {np.mean(accs)}")

        weights = [x.weight.clone().detach().cpu() for x in net.conv]
        if plot:
            fig, ax = plt.subplots(1, 2, figsize = (7, 14))
            weight, initial_weight = [], []

            max_ks = max(kernel_sizes)

            for i, (initial_weight_i, weight_i) in enumerate(zip(initial_weights, weights)):
                if i > 0:
                    initial_weight_i = initial_weight_i[0, :channels_to_plot, :, :].unsqueeze(1)
                    weight_i = weight_i[0, :channels_to_plot, :, :].unsqueeze(1)
                # pad to max kernel size
                initial_weight_i = torch.nn.functional.pad(initial_weight_i, 
                                                           (0, max_ks - initial_weight_i.shape[-1], 
                                                            0, max_ks - initial_weight_i.shape[-2]))
                weight_i = torch.nn.functional.pad(weight_i,
                                                   (0, max_ks - weight_i.shape[-1], 
                                                    0, max_ks - weight_i.shape[-2]))
                approx_sqrt = int(np.sqrt(initial_weight_i.shape[0]))
                initial_weight_i = make_grid(initial_weight_i, 
                                             nrow = approx_sqrt)
                weight_i = make_grid(weight_i,
                                    nrow = approx_sqrt)
                initial_weight.append(initial_weight_i)
                weight.append(weight_i)
            initial_weight = torch.cat(initial_weight, dim = 1)
            weight = torch.cat(weight, dim = 1)

            ax[0].imshow(initial_weight.permute(1, 2, 0))
            ax[1].imshow(weight.permute(1, 2, 0))
            ax[0].set_title("Initial weights")
            ax[1].set_title("Final weights")
            plt.show()
        return net

if __name__ == "__main__":

    #test_simple(pca = False)
    conv = test_conv()
