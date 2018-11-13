import sys
sys.path.append("../../..")

import gym
from gym import error, spaces

from ai_safety_gridworlds.environments import side_effects_sokoban
from ai_safety_gridworlds.environments.shared.safety_game import Actions
import numpy as np


import logging
logger = logging.getLogger(__name__)

class SideEffectsSokoban(gym.Env):
    """
    SideEffectsSokoban Env
    """
    def __init__(self):

        self.env = side_effects_sokoban.SideEffectsSokobanEnvironment()
        self.actions_dict = {'l': Actions.LEFT.value, 'r': Actions.RIGHT.value,
                         'u': Actions.UP.value, 'd': Actions.DOWN.value}
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.s = self.env.reset()
        self.time_step = self.s
        return self.s.observation['board']

    def step(self, a):
        self.time_step = self.env.step(a)
        observation = self.time_step.observation['board']
        reward = self.time_step.reward
        if reward == None:
            print(observation,a, self.time_step.discount)
        if self.time_step.discount == 0.0:
            done = True
            if reward == -1:
                reward = -50
            else:
                reward = 50
        else:
            done = False
        info = self.time_step.observation['extra_observations']
        info['hidden_reward'] = self.env._get_hidden_reward()
        return observation, reward, done, info


    def render(self):
        img = self.time_step.observation['RGB']
        img = np.asarray(img).astype(np.uint8)
        img = np.moveaxis(img, 0, -1)
        return img
