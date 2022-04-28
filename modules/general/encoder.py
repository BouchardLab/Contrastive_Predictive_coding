import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder module, consiting of Convolutional blocks with ReLU activation layers
    Add a convolutional block for each stride / filter (kernel) / padding.
    """

    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()

        self.seq = nn.Sequential()

        self.seq.add_module("linear layer", nn.Linear(input_dim, hidden_dim))

    def forward(self, x):
        return self.seq(x)
