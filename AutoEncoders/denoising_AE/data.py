import pandas as pd
import torch as T
import torch.distributions.normal as normal
import torch.nn.functional as F

class Data:
    def __init__(self, dir):
        self.dir = dir

    def import_data(self):
        data = pd.read_csv(self.dir)
        data = self.normalize(data)

        target = data.iloc[:, 1:]
        input = data.iloc[:, 1:]
        target = T.Tensor(target.values)
        input = T.Tensor(input.values)
        target = T.Tensor(self.add_noise(target))
        return target, input

    def normalize(self, data):
        return data/255.00

    def add_noise(self, data):
        norm = normal.Normal(0,0.1)
        noise = norm.sample(data.shape)

        data = data + noise

        return F.relu(data)