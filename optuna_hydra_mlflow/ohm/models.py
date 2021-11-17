import torch.nn as nn
import torch.nn.functional as F


class Mini(nn.Module):
    def __init__(self, nb_hidden, **cfg):
        super().__init__()
        self.fc1 = nn.Linear(28*28, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_hidden)
        self.fc3 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Default(nn.Module):
    def __init__(self, nb_hidden, nb_layers, **cfg):
        super().__init__()

        self.fcs = []
        self.fcs.append(nn.Linear(28*28, nb_hidden))
        for _ in range(nb_layers - 2):
            self.fcs.append(nn.Linear(nb_hidden, nb_hidden))
        self.fcs.append(nn.Linear(nb_hidden, 10))
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        for l in self.fcs[:-1]:
            x = F.relu(l(x))
        x = self.fcs[-1](x)
        return x
