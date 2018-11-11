import torch as T
from torch.autograd import Variable
import numpy as np

class Learner(object):
    def __init__(self, CNN, input, target, batch_size=1000, epochs=1000):
        self.CNN = CNN
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
        self.cpu_device = T.device('cpu')

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

    def evaluate(self, sample_size=4000):
        rndm = np.random.randint(0, self.input.shape[0]-sample_size-1, 1)
        predictions = self.CNN.forward(self.input[rndm[0]:rndm[0]+sample_size])
        pred = T.argmax(predictions, dim=1)
        result = T.sum((pred == self.target[rndm[0]:rndm[0]+sample_size]))
        print('INTERMEDIATE ACCURACY (', str(rndm[0]), '~', str(rndm[0]+sample_size-1),') : ', str(np.round(result.item()/sample_size * 100.0, 3)), '%')

    def learn(self):
        for e in range(self.epochs) :
            for b in range(self.batch_num_total-1):
                self.CNN.optimizer.zero_grad()
                predictions = self.CNN.forward(self.current_batch)
                loss = self.CNN.loss(predictions, self.current_target)
                print('epoch', str(e+1), '[ batch', str(b+1),'] - loss : ', str(loss.item()))
                loss.backward()
                self.next_batch()
                self.CNN.optimizer.step()
            self.evaluate(sample_size=10000)
