import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DNN(nn.Module):
    def __init__(self, lr):
        super(DNN, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(784, 1024, bias=True), nn.Dropout(0.3))
        self.bn1 = nn.BatchNorm1d(1024)
        self.layer2 = nn.Sequential(nn.Linear(1024, 1200, bias=True), nn.Dropout(0.3))
        self.bn2 = nn.BatchNorm1d(1200)
        self.layer3 = nn.Sequential(nn.Linear(1200, 1024, bias=True), nn.Dropout(0.3))
        self.bn3 = nn.BatchNorm1d(1024)
        self.layer4 = nn.Sequential(nn.Linear(1024,512, bias=True), nn.Dropout(0.3))
        self.bn4 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, 10)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        layer1 = self.bn1(F.relu(self.layer1(data)))
        layer2 = self.bn2(F.relu(self.layer2(layer1)))
        layer3 = self.bn3(F.relu(self.layer3(layer2)))
        layer4 = self.bn4(F.relu(self.layer4(layer3)))
        out = self.out(layer4)
        return F.softmax(out, dim=1)