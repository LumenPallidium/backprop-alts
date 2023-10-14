import torch
import einops

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


if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt

    test_simple(pca = False)
