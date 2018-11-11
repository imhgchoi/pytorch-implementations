import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self, lr):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 16, 4, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.fc1 = nn.Sequential(nn.Linear(16*2*2, 256, bias=True), nn.Dropout(0.2))
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Sequential(nn.Linear(256, 128, bias=True), nn.Dropout(0.2))
        self.bn5 = nn.BatchNorm1d(128)
        self.out = nn.Linear(128, 10)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        conv1 = self.bn1(F.relu(self.conv1(data)))
        pool1 = F.max_pool2d(conv1, kernel_size=(2,2))
        conv2 = self.bn2(F.relu(self.conv2(pool1)))
        pool2 = F.max_pool2d(conv2, kernel_size=(2,2))
        conv3 = self.bn3(F.relu(self.conv3(pool2)))
        pool3 = F.max_pool2d(conv3, kernel_size=(2,2))

        pool3 = pool3.view(-1,16*2*2)

        fc1 = self.bn4(F.relu(self.fc1(pool3)))
        fc2 = self.bn5(F.relu(self.fc2(fc1)))
        out = self.out(fc2)
        return F.softmax(out, dim=1)