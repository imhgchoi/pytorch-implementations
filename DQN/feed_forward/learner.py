import torch as T
from torch.autograd import Variable
import matplotlib.pyplot as plt


class Learner(object):
    def __init__(self, env, agent, episode, target_update):
        self.env = env
        self.agent = agent

        self.episode = episode
        self.target_update = target_update

        self.scores = []

    def initialize_memory(self):
        print('filling up memory...')
        while self.agent.current_memory_no <= self.agent.max_memory:
            state = T.Tensor([self.env.reset()]).cuda()
            done=False
            count=0
            while not done :
                count += 1
                action = self.agent.select_action(state)
                next_state, reward, done, info = self.env.step(int(action.item()))
                next_state = T.Tensor([next_state]).cuda()

                if done and count <= 300:
                    reward = -100

                sars = [state, action, reward, next_state]
                self.agent.memorize(sars)

                state = next_state
        print('memory initialization completed')


    def learn(self):
        self.initialize_memory()
        for e in range(self.episode):
            done = False
            count = 0
            score = 0
            state = T.Tensor([self.env.reset()]).cuda()
            while not done :
                count += 1
                action = self.agent.select_action(state)

                next_state, reward, done, info = self.env.step(int(action.item()))
                next_state = T.Tensor([next_state]).cuda()

                score += reward
                if done and count <= 300 :
                    reward = -100

                sars = [state, action, reward, next_state]
                self.agent.memorize(sars)

                state = next_state

                # start training policy network
                batch_sars = self.agent.recall()

                former_state_cat = T.stack([arr[0] for arr in batch_sars], dim=0).squeeze(1)
                action_cat = [arr[1] for arr in batch_sars]
                reward_cat = T.Tensor([arr[2] for arr in batch_sars]).cuda()
                next_state_cat = T.stack([arr[3] for arr in batch_sars], dim=0).squeeze(1)

                q_pred = self.agent.predictNN.forward(former_state_cat)
                predictions = T.stack([q_pred[i][int(act.item())] for i, act in enumerate(action_cat)]).cuda()

                q_target = self.agent.targetNN.forward(next_state_cat).max(1)[0]
                q_target = reward_cat + self.agent.gamma * q_target
                q_target = Variable(q_target.cuda())

                loss = self.agent.predictNN.loss(predictions, q_target)

                self.agent.predictNN.optimizer.zero_grad()
                print('episode {}'.format(e),'[time step {}] : '.format(count),'loss = {}'.format(loss))
                loss.backward()
                self.agent.predictNN.optimizer.step()

            print('episode {}'.format(e),'score : {}'.format(score))
            if e % self.target_update == 0:
                self.agent.update_targetNN()
                print('target neural net updated')

            self.scores.append(score)
            self.plot_score()

        self.env.env.render()
        self.env.env.close()

    def plot_score(self):
        plt.plot(self.scores)
        plt.xlabel('epsiode')
        plt.ylabel('score')
        plt.title('feed forward DQN training scores')
        plt.savefig('./scores.png')
        plt.close()

