import torch
from torch import nn


class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim):
        super(MultilayerPerceptron, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.register_buffer('threshold', torch.tensor(0.5))

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x.squeeze()
