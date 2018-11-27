import gym
import torch as T
import torchvision.transforms as trans
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Environment(object):
    def __init__(self):
        self.env = gym.make('CartPole-v0').unwrapped

        # https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#sphx-glr-download-intermediate-reinforcement-q-learning-py
        self.resize = trans.Compose([trans.ToPILImage(), trans.Resize(40, interpolation=Image.CUBIC), trans.ToTensor()])
        self.screen_width = 600

    def reset(self):
        self.env.reset()
        last_obs = T.mean(self.get_screen(), dim=1).unsqueeze(1)
        current_obs = T.mean(self.get_screen(), dim=1).unsqueeze(1)
        return last_obs, current_obs

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        observation = T.mean(self.get_screen(), dim=1).unsqueeze(1)
        return observation, reward, done, info

    def get_cart_location(self):
        #https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#sphx-glr-download-intermediate-reinforcement-q-learning-py
        world_width = self.env.x_threshold * 2
        scale = self.screen_width / world_width
        return int(self.env.state[0] * scale + self.screen_width / 2.0)

    def get_screen(self):
        #https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#sphx-glr-download-intermediate-reinforcement-q-learning-py
        """
        In order to prevent the game window from popping up when extracting the rgb array,
        I changed the following line in site-packages/gym/envs/classic_control/rendering.py Viewer class as follows :

        self.window = pyglet.window.Window(width=width, height=height, display=display)
        TO
        self.window = pyglet.window.Window(width=width, height=height, display=display, visible=False)
        """
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        screen = screen[:, 160:320]
        view_width = 320
        cart_location = self.get_cart_location()
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (self.screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescare, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = T.from_numpy(screen)
        # Resize, and add a batch dimension (BCHW)
        return self.resize(screen).unsqueeze(0).to(T.device('cuda' if T.cuda.is_available() else 'cpu'))
