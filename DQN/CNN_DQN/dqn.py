import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, lr=0.001):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, stride=1)
        self.b1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=1)
        self.b2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 5, stride=1)
        self.b3 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(448, 128)
        self.out = nn.Linear(128, 2)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        print(self.device)
        self.to(self.device)

    def forward(self, input):
        conv1 = F.relu(self.b1(self.conv1(input)))
        pool1 = F.max_pool2d(conv1, kernel_size=(2,2))
        conv2 = F.relu(self.b2(self.conv2(pool1)))
        pool2 = F.max_pool2d(conv2, kernel_size=(2,2))
        conv3 = F.relu(self.b3(self.conv3(pool2)))
        pool3 = F.max_pool2d(conv3, kernel_size=(2,2))

        flattened = pool3.view(-1, 448)

        fc1 = F.relu(self.fc1(flattened))
        out = self.out(fc1)

        return out





