from CNN_DQN.environment import Environment
from CNN_DQN.agent import Agent
from CNN_DQN.learner import Learner

if __name__ == '__main__' :
    LEARNING_RATE = 0.005
    MAX_MEMORY = 10000
    BATCH_SIZE = 300
    EPISODE = 50000
    TARGET_UPDATE = 20
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    GAMMA = 0.999

    env = Environment()
    agent = Agent(LEARNING_RATE, MAX_MEMORY, BATCH_SIZE, EPS_START, EPS_END, EPS_DECAY, GAMMA)

    learner = Learner(env, agent, EPISODE, TARGET_UPDATE)
    learner.learn()