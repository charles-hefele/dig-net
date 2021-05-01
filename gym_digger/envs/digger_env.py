import sys
import numpy as np
import gym
from gym import spaces
from gym.utils import colorize
from io import StringIO

# BATTERY_LIFE = 200    # good for 2x2 and 3x3
BATTERY_LIFE = 10000

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
DIG = 4

MAPS = {
    '2x2_a': np.array([
        [1, 1],
        [1, 1]
    ]),
    '2x2_b': np.array([
        [2, 1],
        [1, 2]
    ]),
    '2x2_c': np.array([
        [3, 1],
        [0, 2]
    ]),
    '2x2_d': np.array([
        [3, 4],
        [4, 2]
    ]),
    '2x2_e': np.array([
        [3, 5],
        [4, 2]
    ]),
    '2x2_f': np.array([
        [3, 6],
        [4, 2]
    ]),
    '2x2_g': np.array([
        [3, 7],
        [4, 2]
    ]),
    '2x2_h': np.array([
        [3, 8],
        [4, 2]
    ]),
    '2x2_i': np.array([
        [3, 9],
        [4, 2]
    ]),
    '2x2_j': np.array([
        [3, 10],
        [4, 2]
    ]),
    '2x2_k': np.array([
        [3, 25],
        [4, 2]
    ]),
    '3x3_a': np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ]),
    '3x3_b': np.array([
        [0, 2, 1],
        [0, 1, 3],
        [2, 3, 2]
    ]),
    '4x4_a': np.array([
        [2, 2, 1, 3],
        [1, 0, 0, 1],
        [2, 0, 1, 1],
        [1, 3, 1, 0]
    ]),
    '10x10': np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 3, 2, 0, 0, 0, 4, 0, 0],
        [0, 0, 1, 2, 0, 0, 4, 5, 4, 0],
        [0, 0, 0, 0, 0, 0, 0, 4, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 2, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 3, 2, 0],
        [0, 1, 2, 2, 0, 0, 3, 3, 2, 0],
        [0, 0, 3, 3, 0, 0, 0, 2, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}


class DiggerEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_name):
        self.nutrients_orig = MAPS[map_name]

        # create a copy to decrement in the simulation
        self.nutrients = np.copy(self.nutrients_orig)

        # set size (assume square)
        self.dim = self.nutrients.shape[0]  # length of one dimension of the matrix
        self.size = self.nutrients.size  # length of the entire matrix

        # game variables
        self.battery = BATTERY_LIFE
        self.score = 0

        # init player loc
        self.row = 0  # top-most cell
        self.col = 0  # left-most cell

        # check for highest value
        highest_nutrient_val = self.nutrients.max()
        highest_robot_pos = self.size - 1
        high = max(highest_nutrient_val, highest_robot_pos)

        # define spaces
        shape = self.size + 1  # the flattened nutrient values grid plus one more slot for the robot position, also flattened
        self.observation_space = spaces.Box(low=0, high=high, shape=(shape,), dtype=np.int)
        self.action_space = spaces.Discrete(5)  # left, down, right, up, dig

        # define vars
        self.last_action = None

    def step(self, action):
        reward = 0

        if action == LEFT:
            self.battery -= 1
            reward = -1
            if self.col > 0:
                self.col -= 1

        if action == DOWN:
            self.battery -= 1
            reward = -1
            if self.row < self.nutrients.shape[0] - 1:
                self.row += 1

        if action == RIGHT:
            self.battery -= 1
            reward = -1
            if self.col < self.nutrients.shape[1] - 1:
                self.col += 1

        if action == UP:
            self.battery -= 1
            reward = -1
            if self.row > 0:
                self.row -= 1

        if action == DIG:
            self.battery -= 1
            if self.nutrients[self.row][self.col] > 0:
                self.nutrients[self.row][self.col] -= 1
                self.score += 1
                reward = 1
            else:
                reward = -1

        # check done conditions
        done = False
        if self.battery == 0 or self.nutrients.sum() == 0:
            done = True

        # update values
        self.last_action = action

        # return observation, reward, done, info
        return self.bundle_observation(), reward, done, {'battery': self.battery}

    def reset(self):
        self.last_action = None
        self.battery = BATTERY_LIFE
        self.row = 0  # top-most cell
        self.col = 0  # left-most cell
        self.nutrients = np.copy(self.nutrients_orig)
        return self.bundle_observation()

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
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

    def robot_pos(self):
        return self.row * self.dim + self.col

    def bundle_observation(self):
        raveled = self.nutrients.ravel()
        return np.append(raveled, self.robot_pos())
