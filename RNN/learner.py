import torch as T
from torch.autograd import Variable
import numpy as np

class Learner(object):
    def __init__(self, RNN, input, target, time_step=10, batch_size=1000, epochs=1000):
        self.RNN = RNN
        self.time_step = time_step
        self.input = Variable(input)
        self.input_dim = input.shape
        self.target = Variable(target)
        self.target_dim = target.shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.current_batch = self.input[:,:self.batch_size,:]
        self.current_target = self.target[:self.batch_size]
        self.batch_num = 0
        self.batch_num_total = self.input_dim[1] // self.batch_size + (self.input_dim[1] % self.batch_size != 0)
        self.cpu_device = T.device('cpu')

    def reset(self):
        self.batch_num = 0
        self.current_batch = self.input[:,:self.batch_size,:]
        self.current_target = self.target[:self.batch_size]

    def next_batch(self):
        self.batch_num += 1
        if self.batch_num == 3 :
            self.current_batch = self.input[:,self.batch_size * self.batch_num : ,:]
            self.current_target = self.target[self.batch_size * self.batch_num : ]
        else :
            self.current_batch = self.input[:,self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1),:]
            self.current_target = self.target[self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1)]
        if self.batch_num == self.batch_num_total :
            self.reset()

    def evaluate(self, sample_size=10):
        rndm = np.random.randint(0, self.input.shape[1]-sample_size-1, 1)
        predictions = self.RNN.forward(self.input[:,rndm[0]:rndm[0]+sample_size,:], sample_size)
        predictions = np.int8((predictions > 0.5))
        target = np.int8(self.target[rndm[0]:rndm[0]+sample_size])
        result = np.sum(np.int8((predictions == target)))
        print('INTERMEDIATE ACCURACY (', str(rndm[0]), '~', str(rndm[0]+sample_size-1),') : ', str(np.round(result.item()/sample_size * 100.0, 3)), '%')

    def learn(self):
        for e in range(self.epochs) :
            for b in range(self.batch_num_total):
                self.RNN.optimizer.zero_grad()
                try :
                    predictions = self.RNN.forward(self.current_batch, self.batch_size)
                except RuntimeError :
                    predictions = self.RNN.forward(self.current_batch, self.input.shape[1] % self.batch_size)
                loss = self.RNN.loss(predictions, self.current_target)
                print('epoch', str(e+1), '[ batch', str(b+1),'] - loss : ', str(loss.item()))
                loss.backward()
                self.next_batch()
                self.RNN.optimizer.step()
            self.evaluate(sample_size=4000)
