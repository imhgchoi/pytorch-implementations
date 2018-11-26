import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, lr=0.001):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 24)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(24, 24)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.out = nn.Linear(24, 2)
        nn.init.xavier_uniform_(self.out.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        layer1 = F.relu(self.fc1(input))
        layer2 = F.relu(self.fc2(layer1))
        out = self.out(layer2)
        return out

