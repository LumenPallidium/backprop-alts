from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from time import time
from torchvision import datasets, transforms

def _prepare_for_epochs():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    transforms.Lambda(lambda x: x.view(-1))])

    mnist = datasets.MNIST('../data', train=True, download=True,
                            transform=transform)
    mnist_val = datasets.MNIST('../data', train=False, download=True,
                            transform=transform)

    accs = []
    errors = []

    return mnist, mnist_val, accs, errors

def mnist_test(net,
               batch_size = 256, 
               n_epochs = 3, 
               n_labels = 10, 
               save_every = 10,
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    details = {"epoch_accs" : [],
               "epoch_times" : [],
               "epoch_samples" : [],}
    
    net.to(device)

    with torch.no_grad():

        mnist, mnist_val, accs, errors = _prepare_for_epochs()

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
            details["epoch_accs"].append(np.mean(epoch_accs))

            accs.extend(epoch_accs)

            start = time()
            for i, (x, y) in tqdm(enumerate(train_loader)):
                x = x.to(device)
                y = y.to(device)
                y = torch.nn.functional.one_hot(y, num_classes = n_labels)

                error = net.train_step(x, y)

                if i % save_every == 0:
                    errors.append(error[0])
            epoch_time = time() - start

            details["epoch_times"].append(epoch_time)
            details["epoch_samples"].append(i * batch_size)

        return accs, errors, y, details
    
def atari_assault_test(net,
                       env_name = "AssaultNoFrameskip-v4",
                       n_epochs = 3,
                       n_labels = 10,
                       save_every = 10,
                       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    details = {"epoch_accs" : [],
               "epoch_times" : [],
               "epoch_samples" : [],}
    
    net.to(device)

    with torch.no_grad():

        env = gym.make(env_name)
        env.reset()
        accs = []
        errors = []

        for epoch in range(n_epochs):
            epoch_accs = []
            start = time()
            for i in tqdm(range(1000)):
                x = env.render(mode = "rgb_array")
                x = torch.tensor(x, dtype = torch.float32)
                x = x.permute(2, 0, 1)
                x = x.unsqueeze(0)
                x = x.to(device)

                #TODO

class ActorPerciever(torch.nn.Module):
    def __init__(self,
                 encoder_final_hw = 20,
                 latent_dim = 64,
                 encoder_depth = 5,
                 actor_depth = 3,
                 perciever_depth = 3,
                 activation = torch.nn.GELU(),
                 in_channels = 3,
                 n_actions = 7,
                 perciever_temp = 0.1,
                 actor_temp = 0.4):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder_depth = encoder_depth
        self.encoder_final_hw = encoder_final_hw
        self.actor_depth = actor_depth
        self.perciever_depth = perciever_depth
        self.activation = activation
        self.softmax = torch.nn.Softmax(dim = -1)
        self.in_channels = in_channels
        self.n_actions = n_actions
        self.perciever_temp = perciever_temp
        self.actor_temp = actor_temp

        mult_per_layer = int(np.exp(np.log(latent_dim / in_channels) / encoder_depth))
        in_outs = [in_channels] + [in_channels * mult_per_layer ** i for i in range(1, encoder_depth)] + [latent_dim]

        self.norm = torch.nn.LayerNorm(in_channels)
        encoder = [torch.nn.Conv2d(in_outs[i], in_outs[i + 1], kernel_size = 3, stride = 2) for i in range(encoder_depth)]
        self.final_encoder = torch.nn.Linear(encoder_final_hw, 1)
        self.encoder = torch.nn.ModuleList(encoder)

        actor = [torch.nn.Linear(latent_dim, latent_dim) for i in range(actor_depth - 1)]
        self.final_act = torch.nn.Linear(latent_dim, n_actions)
        self.actor = torch.nn.ModuleList(actor)

        perciever = [torch.nn.Linear(latent_dim, latent_dim) for i in range(perciever_depth)]
        self.perciever = torch.nn.ModuleList(perciever)

        self.efference_table = torch.nn.Embedding(n_actions, latent_dim)

    def encode(self, x):
        x = self.norm(x).permute(0, 3, 1, 2)
        for layer in self.encoder:
            x = layer(x)
            x = self.activation(x)
        x = x.view(-1, self.latent_dim, self.encoder_final_hw)
        x = self.final_encoder(x).squeeze(-1)
        return self.softmax(x)
    
    def act(self, x):
        for layer in self.actor:
            x = layer(x)
            x = self.activation(x)
        act = self.final_act(x)
        act = self.softmax(act / self.actor_temp)
        return act
    
    def perceive(self, x, efference = None):
        if efference is not None:
            act = torch.argmax(efference, dim = -1)
            x = x + self.efference_table(act)
        for layer in self.perciever:
            x = layer(x)
            x = self.activation(x)
        x = self.softmax(x / self.perciever_temp)
        return x
    
    def forward(self, x):
        encoding = self.encode(x)
        action = self.act(encoding)
        perception = self.perceive(encoding, efference = action)
        return action, perception
    
    def train_step(self, x_tminus, env, optimizer_p, optimizer_a):
        optimizer_p.zero_grad()
        optimizer_a.zero_grad()

        action, perception = self(x_tminus)

        xtplus, reward, done, _, _ = env.step(torch.argmax(action, dim = -1))
        xtplus = torch.tensor(xtplus, dtype = torch.float32)
        perception_plus = self.encode(xtplus.unsqueeze(0))

        loss_p = torch.nn.functional.cross_entropy(perception, perception_plus)
        loss_a = -loss_p

        loss_p.backward(retain_graph = True)
        optimizer_p.step()

        loss_a.backward()
        optimizer_a.step()
        return xtplus, loss_p.item()


if __name__ == "__main__":
    from itertools import chain
    ap = ActorPerciever()
    optimizer_p = torch.optim.Adam(chain(ap.encoder.parameters(), ap.perciever.parameters()), lr = 0.001)
    optimizer_a = torch.optim.Adam(chain(ap.encoder.parameters(), ap.actor.parameters()), lr = 0.001)

    env = gym.make("AssaultNoFrameskip-v4")
    x, start_md = env.reset()
    x = torch.Tensor(x).unsqueeze(0)
    losses = []
    for i in tqdm(range(1000)):
        x, loss = ap.train_step(x, env, optimizer_p, optimizer_a)
        losses.append(loss)
    plt.plot(losses)