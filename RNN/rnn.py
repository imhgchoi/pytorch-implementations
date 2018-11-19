import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class RNN(nn.Module):
    # Largely referenced the following page :
    # http://www.jessicayung.com/lstms-for-time-series-in-pytorch/
    def __init__(self, lr):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(1, 20, 2, dropout=0.1)
        self.out = nn.Linear(20,1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input, batch_size):
        lstm_out, self.hidden = self.lstm(input)
        y_pred = T.sigmoid(self.out(lstm_out[-1].view(batch_size, -1)))
        return y_pred.view(-1)




