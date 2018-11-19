import pandas as pd
import torch as T

class Data:
    """
    Here, we generate input data with 120 time steps, which looks into the future of 30 days.
    That is, with the past 120-day data, we attempt to predict whether the return of stock will increase or decrease
    after 30 days
    """
    def __init__(self, dir):
        self.dir = dir
        self.time_step = 120
        self.how_far = 30

    def import_data(self):
        data = pd.read_csv(self.dir)
        input, target = self.preprocess(data)
        input = self.reshape(input)
        return input, target

    def preprocess(self, data):
        data = data['Adj. Close']
        input = list()
        target = list()
        for i in range(data.shape[0]-self.time_step-self.how_far) :
            time_series = list(data.iloc[i:i+self.time_step])
            adj_ts = time_series/time_series[0]
            input.append(adj_ts)
            target.append(int(data.iloc[i+self.time_step+self.how_far]/time_series[0]>adj_ts[-1]))

            if i%500 == 0 :
                print('item',str(i+1),'preprocessed')
        return T.Tensor(input), T.Tensor(target)

    def reshape(self, data):
        return data.transpose(0,1).unsqueeze(2)