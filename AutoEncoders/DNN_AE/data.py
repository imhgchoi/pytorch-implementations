import pandas as pd
import torch as T

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
        return target, input

    def normalize(self, data):
        return data/255.00