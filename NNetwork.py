import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, first_hidden_size, second_hidden_size, obs_size=2):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, first_hidden_size),
            nn.ReLU(),
            nn.Linear(first_hidden_size, second_hidden_size),
            nn.ReLU(),
            nn.Linear(second_hidden_size, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)
