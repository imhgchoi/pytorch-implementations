from feed_forward_DQN.dqn import DQN
import torch as T
import random
import math

class Agent(object):
    def __init__(self, lr, max_mem, bs, eps_start, eps_end, eps_decay, gamma):
        self.lr = lr
        self.targetNN = DQN()
        self.predictNN = DQN(lr)

        self.max_memory = max_mem
        self.current_memory_no = 0
        self.memory = list()
        self.batch_size = bs

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.epsilon = eps_start
        self.gamma = gamma
        self.step = 0

    def reset(self):
        self.targetNN = DQN()
        self.predictNN = DQN(self.lr)

        self.current_memory_no = 0
        self.memory = list()

        self.epsilon = self.eps_start
        self.step = 0

    def update_targetNN(self):
        self.targetNN.load_state_dict(self.predictNN.state_dict())

    def select_action(self, state):
        #https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        #epsilon = self.eps_end + (self.eps_start-self.eps_end)*math.exp(-1 * self.step/self.eps_decay)
        if self.epsilon > self.eps_end :
            self.epsilon = self.epsilon * self.eps_decay
        if random.random() > self.epsilon :
            with T.no_grad() :
                action = self.predictNN.forward(state)[0].max(0)[1].unsqueeze(0).unsqueeze(0).cpu()
                return action
        else :
            return T.Tensor([[random.randrange(2)]], device=self.targetNN.device)

    def memorize(self, sars):
        self.current_memory_no += 1
        if self.current_memory_no <= self.max_memory :
            self.memory.append(sars)
        else :
            self.memory[self.current_memory_no % self.max_memory] = sars

    def recall(self):
        if self.current_memory_no < self.batch_size :
            sample = self.memory.copy()
        else :
            sample = random.sample(self.memory, self.batch_size)

        return sample
