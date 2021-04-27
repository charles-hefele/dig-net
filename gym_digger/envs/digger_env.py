import sys
import numpy as np
import gym
from gym import spaces
from gym.utils import colorize
from io import StringIO

class DiggerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 2), dtype=np.int)
        self.action_space = spaces.Discrete(5)
        self.done = False
        self.last_action = None

        # init nutrients
        # self.nutrients = np.array([
        #     [1, 1],
        #     [1, 1]
        # ])

        self.nutrients = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 2, 0, 0, 0, 4, 0, 0],
            [0, 0, 1, 2, 0, 0, 4, 5, 4, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 3, 2, 0],
            [0, 1, 2, 2, 0, 0, 3, 3, 2, 0],
            [0, 0, 3, 3, 0, 0, 0, 2, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        # init grid
        self.grid = [[0 for i in range(self.nutrients.shape[0])] for j in range(self.nutrients.shape[1])]

        # game variables
        self.battery = 10000
        self.score = 0

        # init player loc
        self.row = 0          # top-most cell
        self.col = 0          # left-most cell
        self.grid[self.row][self.col] = 1

    def step(self, action):
        reward = 0

        # left
        if action == 0:
            if self.col > 0:
                self.battery -= 1
                self.grid[self.row][self.col] = 0
                self.col -= 1
                self.grid[self.row][self.col] = 1

        # down
        if action == 1:
            if self.row < self.nutrients.shape[0] - 1:
                self.battery -= 1
                self.grid[self.row][self.col] = 0
                self.row += 1
                self.grid[self.row][self.col] = 1

        # right
        if action == 2:
            if self.col < self.nutrients.shape[1] - 1:
                self.battery -= 1
                self.grid[self.row][self.col] = 0
                self.col += 1
                self.grid[self.row][self.col] = 1

        # up
        if action == 3:
            if self.row > 0:
                self.battery -= 1
                self.grid[self.row][self.col] = 0
                self.row -= 1
                self.grid[self.row][self.col] = 1

        # dig
        if action == 4:
            self.battery -= 1
            if self.nutrients[self.row][self.col] > 0:
                self.nutrients[self.row][self.col] -= 1
                self.score += 1
                reward = 1
            else:
                reward = -1

        # check done conditions
        if self.battery == 0 or self.nutrients.sum() == 0:
            self.done = True

        # update values
        self.last_action = action
        return (self.nutrients, self.row, self.col), reward, self.done, {'battery': self.battery}

    def reset(self):
        self.last_action = None

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        desc = np.asarray(self.nutrients, dtype='c')
        desc = desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[self.row][self.col] = colorize(desc[self.row][self.col], "red", highlight=True)

        if self.last_action is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up", "Dig"][self.last_action]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")