import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import gymnasium as gym
from time import time
from torchvision import datasets, transforms

def losses_to_running_loss(losses, alpha = 0.95):
    running_losses = []
    running_loss = losses[0]
    for loss in losses:
        running_loss = (1 - alpha) * loss + alpha * running_loss
        running_losses.append(running_loss)
    return running_losses

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

def run_val(net, val_loader, n_labels, epoch_accs, device):
    for i, (x, y) in tqdm.tqdm(enumerate(val_loader)):
        x = x.to(device)
        y = y.to(device)
        y_copy = y.clone().detach()
        y = torch.nn.functional.one_hot(y, num_classes = n_labels)

        y_hat = net(x)
        y_out = torch.argmax(y_hat, dim = 1)
        acc = (y_out == y_copy).float().mean()
        epoch_accs.append(acc.item())
    return epoch_accs

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

            epoch_accs = run_val(net,
                                 val_loader,
                                 n_labels,
                                 epoch_accs,
                                 device)

            print(f"Epoch {epoch} Accuracy: {np.mean(epoch_accs)}")
            details["epoch_accs"].append(np.mean(epoch_accs))

            accs.extend(epoch_accs)

            start = time()
            for i, (x, y) in tqdm.tqdm(enumerate(train_loader)):
                x = x.to(device)
                y = y.to(device)
                y = torch.nn.functional.one_hot(y, num_classes = n_labels)

                error = net.train_step(x, y)

                if i % save_every == 0:
                    errors.append(error[0])
                    
            epoch_time = time() - start

            details["epoch_times"].append(epoch_time)
            details["epoch_samples"].append(i * batch_size)

        # do final validation
        epoch_accs = []
        epoch_accs = run_val(net,
                             val_loader,
                             n_labels,
                             epoch_accs,
                             device)
        print(f"Final Accuracy: {np.mean(epoch_accs)}")
        details["epoch_accs"].append(np.mean(epoch_accs))
        accs.extend(epoch_accs)

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
    """
    Agent based on Schmidhuber's artificial curiosity.

    An encoder encodes environment state into a latent space. A perciever
    module attempts to predict the next state of the environment given the
    current state and an action. An actor module attempts to produce an action
    that maximizes the prediction error of the perciever module.

    Parameters
    ----------
    encoder_final_hw : int
        Height and width of final encoder output.
    latent_dim : int
        Dimensionality of latent space.
    encoder_depth : int
        Number of convolutional layers in encoder.
    actor_depth : int
        Number of linear layers in actor.
    perciever_depth : int
        Number of linear layers in perciever.
    activation : torch.nn.Module
        Activation function to use in all layers.
    in_channels : int
        Number of channels in input image.
    n_actions : int
        Number of actions in environment.
    perciever_temp : float
        Temperature of perciever softmax.
    actor_temp : float
        Temperature of actor softmax.
    """
    def __init__(self,
                 encoder_final_hw = 20,
                 latent_dim = 64,
                 hidden_dim = 64,
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
        self.hidden_dim = hidden_dim
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

        actor = [torch.nn.Linear(latent_dim + hidden_dim, latent_dim + hidden_dim) for i in range(actor_depth - 1)]
        self.final_act = torch.nn.Linear(latent_dim, n_actions)
        self.actor = torch.nn.ModuleList(actor)

        hidden_state = torch.zeros(1, hidden_dim)
        self.register_buffer("actor_hidden_state", hidden_state)

        perciever = [torch.nn.Linear(latent_dim + hidden_dim, latent_dim + hidden_dim) for i in range(perciever_depth)]
        self.perciever = torch.nn.ModuleList(perciever)

        hidden_state = torch.zeros(1, hidden_dim)
        self.register_buffer("perciever_hidden_state", hidden_state)

        self.efference_table = torch.nn.Embedding(n_actions, latent_dim)

    def encode(self, x):
        x = self.norm(x).permute(0, 3, 1, 2)
        for layer in self.encoder:
            x = layer(x)
            x = self.activation(x)
        x = x.view(-1, self.latent_dim, self.encoder_final_hw)
        x = self.final_encoder(x).squeeze(-1)
        return x#self.softmax(x)
    
    def act(self, x):
        x = torch.cat([x, self.actor_hidden_state], dim = -1)
        for layer in self.actor:
            x = layer(x)
            x = self.activation(x)
        act, self.actor_hidden_state = x.split(self.latent_dim, dim = -1)
        act = self.final_act(act)
        act = self.softmax(act / self.actor_temp)
        return act
    
    def perceive(self, x, efference = None):
        if efference is not None:
            x = x + self.efference_table(efference)
        x = torch.cat([x, self.perciever_hidden_state], dim = -1)
        for layer in self.perciever:
            x = layer(x)
            x = self.activation(x)
        #x = self.softmax(x / self.perciever_temp)
        x, self.perciever_hidden_state = x.split(self.latent_dim, dim = -1)
        return x
    
    def forward(self, x, train = True):
        encoding = self.encode(x).detach()
        action = self.act(encoding)
        # sample action based on distribution
        action_val = torch.multinomial(action, 1).squeeze()
        perception = self.perceive(encoding, efference = action_val.detach())
        if train:
            with torch.no_grad():
                perception_a = self.perceive(encoding, efference = action_val)
            return action, perception, perception_a, action_val
        return action, perception, action_val
    
    def train_step(self,
                   x_tminus,
                   env,
                   optimizer_p,
                   optimizer_a,
                   step = False,
                   loss_p = 0,
                   loss_a = 0):
        """
        This train step method ensures compatibility with the non-backprop
        versions to be trained later.
        """
        action, perception, perception_a, action_val = self(x_tminus)

        xtplus, _, done, trunc, _ = env.step(action_val)
        if done or trunc:
            print("Resetting...")
            xtplus, _ = env.reset()
            xtplus = torch.Tensor(xtplus).unsqueeze(0)
            loss_percep = torch.tensor(0, dtype = torch.float32)
            loss_action = torch.tensor(0, dtype = torch.float32)

            self.actor_hidden_state = torch.zeros(1, self.hidden_dim)
            self.perciever_hidden_state = torch.zeros(1, self.hidden_dim)
        else:
            xtplus = torch.Tensor(xtplus).unsqueeze(0)
            perception_plus = self.encode(xtplus)

            loss_percep = loss_p + torch.nn.functional.mse_loss(perception,
                                                                perception_plus)
            loss_action = loss_a - torch.nn.functional.mse_loss(perception_a,
                                                                perception_plus)


            if step:
                optimizer_p.zero_grad()
                optimizer_a.zero_grad()

                loss_percep.backward(retain_graph = True)
                optimizer_p.step()
                loss_action.backward()
                optimizer_a.step()

                self.actor_hidden_state = self.actor_hidden_state.detach()
                self.perciever_hidden_state = self.perciever_hidden_state.detach()
        return xtplus, loss_percep, loss_action, action_val.item()

#TODO : hebbian encoder
#TODO : synthetic gradient
if __name__ == "__main__":
    from gymnasium.wrappers import RecordVideo
    n_steps = 100000
    bptt_steps = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check for mps
    device = torch.device("mps" )

    trigger = lambda t: t % 2 == 0
    ap = ActorPerciever().to(device)
    optimizer_p = torch.optim.Adam(ap.perciever.parameters(), lr = 0.01)
    optimizer_a = torch.optim.Adam(ap.actor.parameters(), lr = 0.02)

    base_env = gym.make("AssaultNoFrameskip-v4", render_mode = "rgb_array")
    env = RecordVideo(base_env, video_folder="../videos", episode_trigger=trigger, disable_logger=True)
    x, _ = env.reset()
    x = torch.Tensor(x).unsqueeze(0)
    losses = []
    pbar = tqdm.trange(n_steps)
    loss_p = 0
    loss_a = 0

    action_histogram = torch.zeros(7)
    for i in range(n_steps):
        opt_step = ((i % bptt_steps) == 0) and (i > 0)
        x, loss_p, loss_a, act = ap.train_step(x,
                                               env,
                                               optimizer_p,
                                               optimizer_a,
                                               step = opt_step,
                                               loss_p = loss_p,
                                               loss_a = loss_a)
        action_histogram[act] += 1
        if opt_step:
            losses.append(loss_p.item())
            pbar.update(bptt_steps)
            pbar.set_description(f"L: {round(loss_p.item(), 4)} A {act}")
            loss_p = 0
            loss_a = 0
    pbar.close()
    
    losses = losses_to_running_loss(losses)
    plt.plot(losses)