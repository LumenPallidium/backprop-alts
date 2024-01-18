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
    entropy_weight : float
        Weight of entropy loss.
    perciever_weight : float
        Weight of perciever loss.
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
                 actor_temp = 0.4,
                 entropy_weight = 0.001,
                 perciever_weight = 1.0):
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
        self.entropy_weight = entropy_weight
        self.perciever_weight = perciever_weight

        mult_per_layer = int(np.exp(np.log(latent_dim / in_channels) / encoder_depth))
        in_outs = [in_channels] + [in_channels * mult_per_layer ** i for i in range(1, encoder_depth)] + [latent_dim]

        self.norm = torch.nn.LayerNorm(in_channels)
        encoder = [torch.nn.Conv2d(in_outs[i], in_outs[i + 1], kernel_size = 3, stride = 2) for i in range(encoder_depth)]
        self.final_encoder = torch.nn.Linear(encoder_final_hw, 1)
        self.encoder = torch.nn.ModuleList(encoder)

        # encoder should not update with gradients
        for param in self.encoder.parameters():
            param.requires_grad = False

        actor = [torch.nn.Linear(latent_dim + hidden_dim, latent_dim + hidden_dim) for i in range(actor_depth - 1)]
        self.final_act = torch.nn.Linear(latent_dim, n_actions)
        self.actor = torch.nn.ModuleList(actor)

        hidden_state = torch.zeros(1, hidden_dim)
        self.register_buffer("actor_hidden_state", hidden_state)

        perciever = [torch.nn.Linear(latent_dim, latent_dim) for i in range(perciever_depth- 1)]
        self.perciever = torch.nn.ModuleList(perciever)

        self.efference_table = torch.nn.Embedding(n_actions, latent_dim)

        # empowerment specific modules
        source = [torch.nn.Linear(latent_dim + hidden_dim, latent_dim + hidden_dim) for i in range(actor_depth - 1)]
        self.final_source = torch.nn.Linear(latent_dim, n_actions)
        self.source = torch.nn.ModuleList(source)

        hidden_state = torch.zeros(1, hidden_dim)
        self.register_buffer("source_hidden_state", hidden_state)

        planner = [torch.nn.Linear(2 * latent_dim, 2 * latent_dim) for i in range(actor_depth - 1)]
        planner.append(torch.nn.Linear(2 * latent_dim, n_actions))
        self.planner = torch.nn.ModuleList(planner)
        
    def run_module(self, x, module, hidden_state = None):
        if hidden_state is not None:
            x = torch.cat([x, hidden_state], dim = -1)
        for layer in module:
            x = layer(x)
            x = self.activation(x)
        if hidden_state is not None:
            x, hidden_state = x.split(self.latent_dim, dim = -1)
            return x, hidden_state
        return x

    def encode(self, x):
        x = self.norm(x).permute(0, 3, 1, 2)
        for layer in self.encoder:
            x = layer(x)
            x = self.activation(x)
        x = x.view(-1, self.latent_dim, self.encoder_final_hw)
        x = self.final_encoder(x).squeeze(-1)
        return x#self.softmax(x)
    
    def act(self, x):
        act, self.actor_hidden_state = self.run_module(x,
                                                       self.actor,
                                                       hidden_state = self.actor_hidden_state)
        act = self.final_act(act)
        act = self.softmax(act / self.actor_temp)
        return act, torch.multinomial(act, 1).squeeze()
    
    def perceive(self, x, efference = None):
        if efference is not None:
            x = x + self.efference_table(efference)
        x = self.run_module(x,
                            self.perciever,
                            hidden_state = None)
        return x
    
    def run_source(self, x):
        x, self.source_hidden_state = self.run_module(x,
                                                      self.source,
                                                      hidden_state = self.source_hidden_state)
        x = self.final_source(x)
        x = self.softmax(x / self.actor_temp)
        return x, torch.multinomial(x, 1).squeeze()
    
    def plan(self, x):
        x = self.run_module(x,
                            self.planner,
                            hidden_state = None)
        x = self.softmax(x / self.actor_temp)
        return x
    
    def forward(self, x):
        encoding = self.encode(x).detach()
        action = self.act(encoding)
        # sample action based on distribution
        action_val = torch.multinomial(action, 1).squeeze()
        return action_val
    
    def train_steps(self,
                    env,
                    last_x,
                    optimizer,
                    bptt_steps):
        """
        This train step method ensures compatibility with the non-backprop
        versions to be trained later.
        """
        optimizer.zero_grad()
        z_t = self.encode(last_x)

        loss = 0
        for i in range(bptt_steps):
            action_probs, action = self.act(z_t)
            source_action_probs, source_action = self.run_source(z_t)

            with torch.no_grad():
                state_pred_source = self.perceive(z_t, efference = source_action)
            plan_action = self.plan(torch.cat([z_t, state_pred_source], dim = -1))

            loss += torch.log((source_action_probs / (plan_action + 1e-8)) + 1e-8).sum()

            # add entropy loss on actor
            entropy = (action_probs * torch.log(action_probs + 1e-8)).sum()
            loss += self.entropy_weight * entropy

            # step the env according to action policy
            xtplus, _, done, trunc, _ = env.step(action)
            xtplus = torch.Tensor(xtplus).unsqueeze(0)
            xtplus = xtplus.to(last_x.device)
            if done or trunc:
                print("Resetting...")
                xtplus, _ = env.reset()
                xtplus = torch.Tensor(xtplus).unsqueeze(0)
                xtplus = xtplus.to(last_x.device)

                self.actor_hidden_state = torch.zeros(1, self.hidden_dim)
                self.source_hidden_state = torch.zeros(1, self.hidden_dim)
            # estimate next state using perciever
            z_t_hat = self.perceive(z_t, efference = action)
            # encode next state
            z_t = self.encode(xtplus)

            loss += torch.nn.functional.mse_loss(z_t_hat, z_t) * self.perciever_weight


        loss /= bptt_steps
        loss.backward()
        optimizer.step()

        self.actor_hidden_state = self.actor_hidden_state.detach()
        self.source_hidden_state = self.source_hidden_state.detach()
        return xtplus, loss, action.item()

#TODO : hebbian encoder
#TODO : synthetic gradient
#TODO : rtrl would be cool
if __name__ == "__main__":
    from gymnasium.wrappers import RecordVideo
    n_steps = 10000
    bptt_steps = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check for mps
    #device = torch.device("mps" if torch.backends.mps.is_available() else device)

    trigger = lambda t: t % 2 == 0
    ap = ActorPerciever().to(device)
    optimizer = torch.optim.Adam(ap.perciever.parameters(), lr = 0.01)

    base_env = gym.make("AssaultNoFrameskip-v4", render_mode = "rgb_array")
    env = RecordVideo(base_env, video_folder="../videos", episode_trigger=trigger, disable_logger=True)
    last_x, _ = env.reset()
    last_x = torch.Tensor(last_x).unsqueeze(0).to(device)
    losses = []
    pbar = tqdm.trange(n_steps)

    for i in range(n_steps):
        last_x, loss, act = ap.train_steps(env,
                                           last_x,
                                           optimizer,
                                           bptt_steps)
        
        losses.append(loss.item())
        pbar.update(1)
        pbar.set_description(f"Loss: {round(loss.item(), 4)}")
    pbar.close()
    
    losses = losses_to_running_loss(losses)
    plt.plot(losses)