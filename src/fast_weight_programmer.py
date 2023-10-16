import torch

class DeltaNet(torch.nn.Module):
    """Variant of Schmidhuber's Fast Weight Programmer (FWP) from this paper: https://arxiv.org/abs/2102.11174"""
    def __init__(self, 
                 dim_in, 
                 dim_out, 
                 dim_hidden,
                 momentum=0.9,):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden

        # TODO: allow functions other than softmax (requirement its output is positive and sums to 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.sigmoid = torch.nn.Sigmoid()

        # slow network, returns k, v, q and a learning rate (shape is d_out + 2 * d_hidden + 1, d_in)
        self.W_slow = torch.nn.Linear(self.dim_in, self.dim_out + 2 * self.dim_hidden + 1)
        # paper reccomends initializing fast networks with zeros
        self.W_fast = torch.nn.Linear(self.dim_hidden, self.dim_out)
        torch.nn.init.constant_(self.W_fast.weight, 0)

    def forward(self, x):
        # get parameters from slow network, matching order in paper
        k, v, q, lr = torch.split(self.W_slow(x), [self.dim_hidden, self.dim_out, self.dim_hidden, 1], dim=1)
        # paper suggests doing this for stability
        lr = self.sigmoid(lr)

        # get fast network output, called v_bar in the paper
        v_bar = self.W_fast(self.softmax(k))

        # this is the weight update from the delta rule
        delta = lr * (v - v_bar).T @ self.sigmoid(k) / x.shape[0]

        # update fast network
        self.W_fast.weight.data += delta

        # output is update fast network acting on query
        return self.W_fast(self.softmax(q))