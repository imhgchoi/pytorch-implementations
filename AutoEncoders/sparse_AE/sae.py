import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as norm
import torch.optim as optim


class SAE(nn.Module):
    def __init__(self, lr, bs):
        super(SAE, self).__init__()
        self.encoder1 = nn.Linear(784, 900)
        self.encoder2 = nn.Linear(900, 1024)

        self.code = nn.Linear(1024, 1300)

        self.decoder1 = nn.Linear(1300, 1024)
        self.decoder2 = nn.Linear(1024, 900)

        self.out = nn.Linear(900, 784)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.batch_size = bs
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, input):
        enc1 = F.relu(self.encoder1(input))
        enc2 = F.relu(self.encoder2(enc1))

        code = self.code(enc2)

        dec1 = F.relu(self.decoder1(code))
        dec2 = F.relu(self.decoder2(dec1))

        out = T.sigmoid(self.out(dec2))

        return code, out


    def sparsity_penalty(self, code):
        rho = T.sigmoid(T.FloatTensor([0.01 for _ in range(code.shape[1])]).unsqueeze(0).cuda())
        rho_hat = T.sum(code, dim=0, keepdim=True)/float(self.batch_size)
        rho_hat = T.sigmoid(rho_hat.cuda())

        return T.sum(rho * T.log(rho/rho_hat)) + T.sum((1-rho) * T.log((1-rho)/(1-rho_hat)))

