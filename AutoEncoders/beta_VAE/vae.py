import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal as norm
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, lr):
        super(VAE, self).__init__()
        self.encoder1 = nn.Linear(784, 512)
        self.encoder2 = nn.Linear(512, 256)
        self.encoder3 = nn.Linear(256,128)

        self.mu = nn.Linear(128, 32)
        self.std = nn.Linear(128, 32)

        self.decoder1 = nn.Linear(32, 128)
        self.decoder2 = nn.Linear(128, 256)
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

        mu = self.mu(enc3)
        logvar = self.std(enc3)

        z = self.reparametrize(mu, logvar)

        dec1 = F.relu(self.decoder1(z))
        dec2 = F.relu(self.decoder2(dec1))
        dec3 = F.relu(self.decoder3(dec2))

        out = T.sigmoid(self.out(dec3))

        return out


    def reparametrize(self, mu, logvar):
        self.mu_z = mu
        self.logvar_z = logvar
        self.std_z = T.exp(logvar * 0.5)
        N = norm.Normal(0, 1)
        eps = N.sample(self.std_z.size()).cuda()

        self.z = self.mu_z + self.std_z * eps

        return self.z


    def latent_loss(self):
        KLD = -0.5 * T.sum(self.logvar_z - self.mu_z.pow(2) - T.exp(self.logvar_z) + 1)
        return KLD