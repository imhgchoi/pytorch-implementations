from torch.autograd import Variable
import numpy as np

class Learner(object):
    def __init__(self, VAE, input, target, batch_size=1000, epochs=1000):
        self.VAE = VAE
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

    def learn(self):
        for e in range(self.epochs) :
            beta = float(np.exp(e)/np.exp(self.epochs))
            for b in range(self.batch_num_total):
                self.VAE.optimizer.zero_grad()
                predictions = self.VAE.forward(self.current_batch)
                loss = self.VAE.loss(predictions, self.current_target) + beta * self.VAE.latent_loss()/(self.batch_size *self.input_dim[1])
                print('epoch', str(e+1), '[ batch', str(b+1),'] - loss : ', str(np.round(loss.item(),7)),' (MSE loss :',str(np.round(loss.item() - beta * self.VAE.latent_loss().item()/(self.batch_size *self.input_dim[1]),5)), '/ latent loss (beta):',str(np.round(self.VAE.latent_loss().item()/(self.batch_size *self.input_dim[1]), 5)),'(',str(beta),'))')
                loss.backward()
                self.next_batch()
                self.VAE.optimizer.step()
        return self.VAE