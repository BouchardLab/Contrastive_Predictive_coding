import torch.nn as nn


def mlp(input_dim, hidden_dim, output_dim, n_layers=1, activation='relu', T=None):
    if activation == 'relu':
        activation_f = nn.ReLU()
    if T is None:
        layers = [nn.Linear(input_dim, hidden_dim), activation_f]
    else:
        layers = [nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(T), activation_f]
    for _ in range(n_layers):
        if T is None:
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_f]
        else:
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(T), activation_f]
    layers += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)


class Encoder_linear(nn.Module):
    """
    Encoder module, consiting of Convolutional blocks with ReLU activation layers
    Add a convolutional block for each stride / filter (kernel) / padding.
    """

    def __init__(self, input_dim, output_dim):
        super(Encoder_linear, self).__init__()

        self.seq = nn.Sequential()

        self.seq.add_module("linear layer", nn.Linear(input_dim, output_dim))

    def forward(self, x):
        return self.seq(x)


class Encoder_nonlinear(nn.Module):
    """
    Encoder module, consiting of Convolutional blocks with ReLU activation layers
    Add a convolutional block for each stride / filter (kernel) / padding.
    """

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder_nonlinear, self).__init__()

        self.seq = nn.Sequential()

        self.seq.add_module("MLP layer", mlp(input_dim, hidden_dim, output_dim))

    def forward(self, x):
        return self.seq(x)
