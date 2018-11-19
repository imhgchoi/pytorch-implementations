import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DNN_AE(nn.Module):
    def __init__(self, lr=0.1):
        super(DNN_AE, self).__init__()
        self.encoder1 = nn.Linear(784, 512)
        self.encoder2 = nn.Linear(512, 256)
        self.encoder3 = nn.Linear(256, 128)

        self.code = nn.Linear(128,64)

        self.decoder1 = nn.Linear(64,128)
        self.decoder2 = nn.Linear(128,256)
        self.decoder3 = nn.Linear(256, 512)

        self.out = nn.Linear(512, 784)


        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        enc1 = F.relu(self.encoder1(input))
        enc2 = F.relu(self.encoder2(enc1))
        enc3 = F.relu(self.encoder3(enc2))

        code = self.code(enc3)

        dec1 = F.relu(self.decoder1(code))
        dec2 = F.relu(self.decoder2(dec1))
        dec3 = F.relu(self.decoder3(dec2))

        out = T.sigmoid(self.out(dec3))

        return out