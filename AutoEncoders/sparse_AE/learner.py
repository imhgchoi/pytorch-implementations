from torch.autograd import Variable
import numpy as np

class Learner(object):
    def __init__(self, SAE, input, target, batch_size=1000, epochs=1000):
        self.SAE = SAE
        self.input = Variable(input)
        self.input_dim = input.shape
        self.target = Variable(target)
        self.target_dim = target.shape
        self.epochs = epochs
        self.batch_size = batch_size
        self.current_batch = self.input[:self.batch_size,:]
        self.current_target = self.target[:self.batch_size,:]
        self.batch_num = 0
        self.batch_num_total = self.input_dim[0] // self.batch_size + (self.input_dim[0] % self.batch_size != 0)

    def reset(self):
        self.batch_num = 0
        self.current_batch = self.input[:self.batch_size,:]
        self.current_target = self.target[:self.batch_size,:]

    def next_batch(self):
        self.batch_num += 1
        self.current_batch = self.input[self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1),:]
        self.current_target = self.target[self.batch_size * self.batch_num : self.batch_size * (self.batch_num+1),:]
        if self.batch_num == self.batch_num_total :
            self.reset()

    def learn(self, beta):
        for e in range(self.epochs) :
            for b in range(self.batch_num_total):
                self.SAE.optimizer.zero_grad()
                code, predictions = self.SAE.forward(self.current_batch)
                sparse_pen = self.SAE.sparsity_penalty(code)
                loss = self.SAE.loss(predictions, self.current_target) + beta * sparse_pen
                print('epoch', str(e+1), '[ batch', str(b+1),'] - loss : ', str(loss.item()),'(sparsity penalty :', str(np.round(sparse_pen.item(), 5)),')')
                loss.backward()
                self.next_batch()
                self.SAE.optimizer.step()
        return self.SAE