import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from einops import rearrange

#TODO : covariance matrix is too big to be allocated
class CMAES(torch.nn.Module):
    def __init__(self,
                 in_dim,
                 pop_size,
                 solution_evaluation_function,
                 out_dim = None,
                 n_layers = 3,
                 dim_mult = 1,
                 step_size_init=0.3,
                 n_steps=1000,
                 top_percent=0.5,
                 alpha = 1.5,
                 cov_rank1_lr = 0.3, 
                 cov_rankmu_lr = 0.3,
                 damping = 1,
                 activation = torch.nn.Tanh(),
                 downsampling = 16,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.in_dim = in_dim
        if out_dim is None:
            out_dim = in_dim
        self.out_dim = out_dim

        self.pop_size = pop_size
        self.top_percent = top_percent
        self.n_layers = n_layers
        self.dim_mult = dim_mult

        self.step_size_init = step_size_init
        self.step_size = step_size_init
        self.n_steps = n_steps
        self.downsampling = downsampling
        self.device = device

        self.layer_sizes, self.block_sizes = self.get_dims()
        print(self.layer_sizes, self.block_sizes)

        self.means = [torch.zeros(dim // self.downsampling, device=device) for dim in self.block_sizes]
        self.covs = [torch.eye(dim // self.downsampling, device=device) for dim in self.block_sizes]

        self.path_steps = [torch.zeros(dim, device=device) for dim in self.block_sizes]
        self.path_covs = [torch.zeros(dim, device=device) for dim in self.block_sizes]

        self.time_horizons = [4 / dim for dim in self.block_sizes]
        self.alpha = alpha
        self.cov_rank1_lr = cov_rank1_lr
        self.cov_rankmu_lr = cov_rankmu_lr
        self.damping = damping
        self.activation = activation

        self.evaluate_solutions = solution_evaluation_function

    def get_dims(self):
        layer_sizes = [int(self.in_dim * (self.dim_mult ** k)) for k in range(self.n_layers)]
        layer_sizes.append(self.out_dim)

        block_sizes = [layer_sizes[i] * layer_sizes[i + 1] for i in range(self.n_layers)]
        return layer_sizes, block_sizes


    def forward(self, x):
        for n in range(self.n_layers):
            # break tensor with shape (pop, product dim) into (pop, dim1, dim2)
            if self.downsampling > 1:
                mean = torch.nn.functional.interpolate(self.means[n][None, None, :],
                                                       size = self.block_sizes[n]).squeeze(0).squeeze(0)
            else:
                mean = self.means[n]
            layer_weight = mean.view(self.layer_sizes[n], self.layer_sizes[n + 1])
            x = torch.einsum('ji,bj->bi', layer_weight, x)
            x = self.activation(x)
        return x
        
    def train_step(self, x, y):
        solutions, rewards = self.sample_solutions(x, y)

        n_chosen = int(self.top_percent * self.pop_size)

        rewards, indices = torch.sort(rewards, descending=True)
        indices = indices[:n_chosen]
        for i in range(self.n_layers):
            mean_copy = self.means[i].clone()
            self.means[i] = self.update_mean(solutions[i], indices)

            delta_mean = self.means[i] - mean_copy
            # note shape is (pop_size, dim) - (dim,) = (pop_size, dim)
            delta_solutions = (solutions[i] - mean_copy) / self.step_size

            self.update_path_step(delta_mean, n_chosen)
            self.update_path_cov(delta_mean, n_chosen)

            self.update_cov(delta_solutions, mean_copy)

            self.update_step_size()

        return rewards

    def sample_solutions(self, x, y):
        x = x.unsqueeze(0).repeat(self.pop_size, 1, 1)
        solutions = []
        
        for i in range(self.n_layers):
            solution = MultivariateNormal(self.means[i], 
                                          self.covs[i]).sample((self.pop_size,))
            full_size = self.block_sizes[i]
            solution = torch.nn.functional.interpolate(solution.unsqueeze(2), 
                                                       size = full_size).squeeze(2)
            # break tensor with shape (pop, product dim) into (pop, dim1, dim2)
            layer_weight = rearrange(solution, 'p(ij)->pij',
                                     i = self.layer_sizes[i],
                                     j = self.layer_sizes[i + 1])
            x = torch.einsum('pij,pbj->pbi', layer_weight, x)
            x = self.activation(x)
            solutions.append(solution)
        
        rewards = self.evaluate_solutions(x, y)
        return solutions, rewards

    @staticmethod
    def update_mean(solutions, indices):
        solutions_kept = solutions[indices]
        # giving equal weight for now, TODO add more strategies
        return torch.mean(solutions_kept, dim=0)

    def update_path_step(self, delta_mean, n_chosen, i):
        diagonal, orthogonal = torch.linalg.eigh(self.covs[i], eigenvectors=True)
        cov_inv_sqrt = orthogonal @ torch.diag(torch.sqrt(1 / diagonal)) @ orthogonal.T

        # this will need be updated if we use different weight strategies TODO
        var_selection_mass = n_chosen

        discount = 1 - self.time_horizons[i]
        discount_comp = np.sqrt(1 - discount ** 2)

        if self.downsampling > 1:
            delta_mean = torch.nn.functional.interpolate(delta_mean.unsqueeze(1), 
                                                         size = self.block_sizes[i]).squeeze(1)
            step_update = torch.nn.functional.interpolate((cov_inv_sqrt @ delta_mean).unsqueeze(1),
                                                          scale_factor = (1 / self.downsampling)).squeeze(1)
        else:
            step_update = cov_inv_sqrt @ delta_mean

        self.path_steps[i] = discount * self.path_steps[i] + discount_comp * np.sqrt(var_selection_mass) * step_update

    def update_path_cov(self, delta_mean, n_chosen, i):
        discount = 1 - self.time_horizons[i]

        if torch.norm(self.path_steps[i]) <= self.alpha * np.sqrt(self.block_sizes[i]):
            path_cov = self.path_covs[i] * discount
        else:
            discount_comp = np.sqrt(1 - discount ** 2)
            path_cov = discount * self.path_covs[i] + discount_comp * np.sqrt(n_chosen) * delta_mean

        self.path_covs[i] = path_cov

    def update_cov(self, delta_solutions, i):
        if torch.norm(self.path_steps[i]**2) >= self.alpha * np.sqrt(self.block_sizes[i]):
            discount_var_loss = self.cov_rank1_lr * (2 - self.time_horizons[i]) * self.time_horizons[i]
        else:
            discount_var_loss = 0
        full_discount = (1 - self.cov_rank1_lr - self.cov_rankmu_lr + discount_var_loss)

        if self.downsampling > 1:
            path_step_ds = torch.nn.functional.interpolate(self.path_steps[i].unsqueeze(1),
                                                            scale_factor = (1 / self.downsampling)).squeeze(1)
            rank1_cov = self.cov_rank1_lr * path_step_ds @ path_step_ds.T

            delta_solutions_ds = torch.nn.functional.interpolate(delta_solutions.unsqueeze(2),
                                                                 scale_factor = (1 / self.downsampling)).squeeze(2)
            ranku_cov = self.cov_rankmu_lr * (delta_solutions_ds @ delta_solutions_ds.T).mean(dim=0)
        else:
            rank1_cov = self.cov_rank1_lr * self.path_steps[i] @ self.path_steps[i].T
            # note mean is weight = 1, TODO fix this if adding more strategies
            ranku_cov = self.cov_rankmu_lr * (delta_solutions @ delta_solutions.T).mean(dim=0)
        ranku_cov /= self.step_size ** 2

        self.covs[i] = full_discount * self.covs[i] + rank1_cov + ranku_cov

    def update_step_size(self, i):
        normal_expectation = np.sqrt(self.block_sizes[i]) * (1 - 1 / (4 * self.block_sizes[i]) + 1 / (21 * self.block_sizes[i] ** 2))
        self.step_size *= np.exp((self.time_horizons[i] / self.damping) * torch.norm(self.path_steps[i]) / normal_expectation - 1)

if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision
    from torchvision import datasets, transforms

    def mnist_solution_evaluation(y_hat, y):
        # y_hat is (pop_size, batch, dim) while y is (batch, dim)
        with torch.no_grad():
            y = y.unsqueeze(0).repeat(y_hat.shape[0], 1, 1)
            # mse loss
            loss = torch.mean((y_hat - y) ** 2, dim = (1, 2))
        # negative because it is reward
        return -loss

    batch_size = 256
    n_epochs = 3
    n_labels = 10
    save_every = 10
    sample_noise_scale = 0.01
    bias = True
    whiten = True
    n_layers = 3
    activation = torch.nn.Tanh()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        transforms.Lambda(lambda x: x.view(-1))])

        mnist = datasets.MNIST('../data', train=True, download=True,
                                transform=transform)
        mnist_val = datasets.MNIST('../data', train=False, download=True,
                                transform=transform)
        
        net = CMAES(784, 100, mnist_solution_evaluation, out_dim = 10, dim_mult = 0.75, activation = activation).to(device)

        accs = []
        errors = []

        for epoch in range(n_epochs):
            train_loader = torch.utils.data.DataLoader(mnist, 
                                                       batch_size = batch_size, 
                                                       shuffle = True)
            val_loader = torch.utils.data.DataLoader(mnist_val,
                                                     batch_size = batch_size,
                                                     shuffle = True)
            epoch_accs = []

            for i, (x, y) in tqdm(enumerate(val_loader)):
                x = x.to(device)
                y = y.to(device)
                y_copy = y.clone().detach()
                y = torch.nn.functional.one_hot(y, num_classes = n_labels)

                y_hat = net(x)
                y_out = torch.argmax(y_hat, dim = 1)
                acc = (y_out == y_copy).float().mean()
                epoch_accs.append(acc.item())

            print(f"Epoch {epoch} Accuracy: {np.mean(epoch_accs)}")

            accs.extend(epoch_accs)

            for i, (x, y) in tqdm(enumerate(train_loader)):
                x = x.to(device)
                y = y.to(device)
                y = torch.nn.functional.one_hot(y, num_classes = n_labels)

                rewards = net.train_step(x, y)

                if i % save_every == 0:
                    errors.append(-rewards[0])
        
        plt.plot(accs)

