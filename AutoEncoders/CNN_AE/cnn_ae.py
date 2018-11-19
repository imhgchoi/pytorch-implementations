import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class CNN_AE(nn.Module):
    def __init__(self, lr=0.1):
        super(CNN_AE, self).__init__()
        self.encoder1 = nn.Conv2d(1, 16, 3, stride=3, padding=1)
        self.encoder2 = nn.Conv2d(16, 8, 3, stride=2, padding=1)

        self.decoder1 = nn.ConvTranspose2d(8, 16, 3, stride=2)
        self.decoder2 = nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1)

        self.out = nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1)


        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        enc1 = F.max_pool2d(F.relu(self.encoder1(input)), kernel_size=(2,2), stride=2)
        code = F.max_pool2d(F.relu(self.encoder2(enc1)), kernel_size=(2,2), stride=1)

        dec1 = F.relu(self.decoder1(code))
        dec2 = F.relu(self.decoder2(dec1))


        out = T.sigmoid(self.out(dec2))

        return out