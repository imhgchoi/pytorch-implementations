import pandas as pd
import torch as T

class Data:
    def __init__(self, dir):
        self.dir = dir

    def import_data(self):
        data = pd.read_csv(self.dir)
        target = data['label']
        input = data.iloc[:, 1:]
        target = T.Tensor(target.values)
        input = T.Tensor(input.values).view(-1,1,28,28)
        return target, input