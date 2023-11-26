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
        Whether or not to include a bias term
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 padding = 0):
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

        self.weight = torch.randn(out_channels, in_channels, kernel_size[0], kernel_size[1])

    def forward(self, x):
        return torch.nn.functional.conv2d(x, self.weight, stride = self.stride, padding = self.padding)
    
    def train_step(self, x, lr = 0.01):
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
                                      
        y = self.forward(x)
        y_unfolded = einops.rearrange(y, "... c h w -> ... c (h w)")

        # sum over batch and image chunks
        dW = torch.einsum("bcnij, bkn -> kcij", x_unfolded, y_unfolded)
        dW /= batch_size * n_blocks

        self.weight -= lr * dW
        return y

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

def test_conv(n_epochs = 2,
              kernel_size = 3,
              stride = 2,
              batch_size = 256,
              mid_channels = 32,
              out_channels = 64,
              lr = 0.01,
              plot = True):
    """
    Function that tests the Hebbian convolution, by plotting the filters.
    """
    from utils import _prepare_for_epochs
    from torchvision.utils import make_grid
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        conv = torch.nn.Sequential(HebbianConv2d(1, mid_channels, kernel_size, stride = stride, padding = 1),
                                   HebbianConv2d(mid_channels, out_channels, kernel_size, stride = stride, padding = 1)).to(device)
        initial_weights = [x.weight.clone().detach().cpu() for x in conv]
        mnist, mnist_val, accs, errors = _prepare_for_epochs()

        for epoch in range(n_epochs):
            train_loader = torch.utils.data.DataLoader(mnist, 
                                                        batch_size = batch_size, 
                                                        shuffle = True)
            for i, (x, y) in tqdm(enumerate(train_loader)):
                x = x.to(device)
                x = x.view(-1, 1, 28, 28)
                y = x.clone()
                for layer in conv:
                    y = layer.train_step(y, lr = lr)

        weights = [x.weight.clone().detach().cpu() for x in conv]
        if plot:
            fig, ax = plt.subplots(1, 2, figsize = (14, 7))
            weight, initial_weight = [], []

            for i, (initial_weight_i, weight_i) in enumerate(zip(initial_weights, weights)):
                if i > 0:
                    initial_weight_i = initial_weight_i[0].unsqueeze(1)
                    weight_i = weight_i[0].unsqueeze(1)
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
        return conv

if __name__ == "__main__":

    #test_simple(pca = False)
    conv = test_conv()
