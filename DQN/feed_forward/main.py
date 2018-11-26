from feed_forward_DQN.environment import Environment
from feed_forward_DQN.agent import Agent
from feed_forward_DQN.learner import Learner

if __name__ == '__main__' :
    LEARNING_RATE = 0.0005
    MAX_MEMORY = 5000
    BATCH_SIZE = 300
    EPISODE = 200
    TARGET_UPDATE = 3
    EPS_START = 0.9
    EPS_END = 0.01
    EPS_DECAY = 0.99
    GAMMA = 0.99

    env = Environment()
    agent = Agent(LEARNING_RATE, MAX_MEMORY, BATCH_SIZE, EPS_START, EPS_END, EPS_DECAY, GAMMA)

    learner = Learner(env, agent, EPISODE, TARGET_UPDATE)
    learner.learn()