import gym

class Environment(object):
    def __init__(self):
        self.env = gym.make('CartPole-v1')

    def reset(self):
        observation = self.env.reset()
        return observation

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info
