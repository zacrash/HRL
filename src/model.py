import torch
import torch.nn as nn
import torch.nn.functional as F


class Q(nn.Module):
    def __init__(self, input_size, action_size, lr):
        # super().__init__()
        super(Q, self).__init__()
        self.lr = lr
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.tensor(x, dtype=torch.float, requires_grad=True)
