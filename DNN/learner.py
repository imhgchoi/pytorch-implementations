import torch as T
from torch.autograd import Variable
import numpy as np

class Learner(object):
    def __init__(self, DNN, input, target, batch_size=1000, epochs=1000):
        self.DNN = DNN
        self.input = Variable(input)
        self.input_dim = input.shape
        self.target = Variable(target).long()
        self.target_dim = target.shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.current_batch = self.input[:self.batch_size,:]
        self.current_target = self.target[:self.batch_size]
        self.batch_num = 0
        self.batch_num_total = self.input_dim[0] // self.batch_size + (self.input_dim[0] % self.batch_size == 0)

    def reset(self):
        self.batch_num = 0
        self.current_batch = self.input[:self.batch_size,:]
        self.current_target = self.target[:self.batch_size]

    def next_batch(self):
        self.batch_num += 1
        self.current_batch = self.input[self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1),:]
        self.current_target = self.target[self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1)]
        if self.batch_num == self.batch_num_total-1 :
            self.reset()

    def evaluate(self):
        predictions = self.DNN.forward(self.input)
        pred = T.argmax(predictions, dim=1)
        result = T.sum((pred == self.target))
        print('INTERMEDIATE ACCURACY : ', str(np.round(result.item()/self.input.shape[0] * 100.0, 3)), '%')

    def learn(self):
        for e in range(self.epochs) :
            for b in range(self.batch_num_total-1):
                self.DNN.optimizer.zero_grad()
                predictions = self.DNN.forward(self.current_batch)
                loss = self.DNN.loss(predictions, self.current_target)
                print('epoch', str(e+1), '[ batch', str(b+1),'] - loss : ', str(loss.item()))
                loss.backward()
                self.next_batch()
                self.DNN.optimizer.step()
            self.evaluate()