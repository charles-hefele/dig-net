import sys
from contextlib import closing

import numpy as np
from io import StringIO

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    0: [
        "11",
        "11"
    ],
    1: [
        "11",
        "10"
    ],
    2: [
        "11",
        "01"
    ],
    3: [
        "11",
        "00"
    ],
    4: [
        "10",
        "11"
    ],
    5: [
        "10",
        "10"
    ],
    6: [
        "10",
        "01"
    ],
    7: [
        "10",
        "00"
    ],
    8: [
        "01",
        "11"
    ],
    9: [
        "01",
        "10"
    ],
    10: [
        "01",
        "01"
    ],
    11: [
        "01",
        "00"
    ],
    12: [
        "00",
        "11"
    ],
    13: [
        "00",
        "10"
    ],
    14: [
        "00",
        "01"
    ],
    15: [
        "00",
        "00"
    ]
}

DIG = [(1, 32, 1, False),   # 0
       (1, 17, 1, False),   # 1
       (1, 10, 1, False),   # 2
       (1, 7, 1, False),    # 3

       (1, 36, 1, False),   # 4
       (1, 21, 1, False),   # 5
       (1, 14, 1, False),   # 6
       (1, 7, -1, False),   # 7

       (1, 40, 1, False),   # 8
       (1, 25, 1, False),   # 9
       (1, 10, -1, False),  # 10
       (1, 15, 1, False),   # 11

       (1, 44, 1, False),   # 12
       (1, 29, 1, False),   # 13
       (1, 14, -1, False),  # 14
       (1, 15, -1, False),  # 15

       (1, 48, 1, False),   # 16
       (1, 17, -1, False),  # 17
       (1, 26, 1, False),   # 18
       (1, 22, 1, False),   # 19

       (1, 52, 1, False),   # 20
       (1, 21, -1, False),  # 21
       (1, 30, 1, False),   # 22
       (1, 23, -1, False),  # 23

       (1, 56, 1, False),   # 24
       (1, 25, -1, False),  # 25
       (1, 26, -1, False),  # 26
       (1, 31, 1, False),   # 27

       (1, 60, 1, True),    # 28
       (1, 29, -1, False),  # 29
       (1, 30, -1, False),  # 30
       (1, 31, -1, False),  # 31

       (1, 32, -1, False),  # 32
       (1, 49, 1, False),   # 33
       (1, 42, 1, False),   # 34
       (1, 39, 1, False),   # 35

       (1, 36, -1, False),  # 36
       (1, 53, 1, False),   # 37
       (1, 46, 1, False),   # 38
       (1, 39, -1, False),  # 39

       (1, 40, -1, False),  # 40
       (1, 57, 1, False),   # 41
       (1, 42, -1, False),  # 42
       (1, 47, 1, False),   # 43

       (1, 44, -1, False),  # 44
       (1, 61, 1, True),    # 45
       (1, 46, -1, False),  # 46
       (1, 47, -1, False),  # 47

       (1, 48, -1, False),  # 48
       (1, 49, -1, False),  # 49
       (1, 58, 1, False),   # 50
       (1, 55, 1, False),   # 51

       (1, 52, -1, False),  # 52
       (1, 53, -1, False),  # 53
       (1, 62, 1, True),    # 54
       (1, 55, -1, False),  # 55

       (1, 56, -1, False),  # 56
       (1, 57, -1, False),  # 57
       (1, 58, -1, False),  # 58
       (1, 63, 1, True),    # 59

       (1, 60, 0, True),    # 60
       (1, 61, 0, True),    # 61
       (1, 62, 0, True),    # 62
       (1, 63, 0, True)]    # 63


class DiggerEnvDiscrete(discrete.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, map_name=0):
        desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = desc.shape

        nA = 5
        total_cells = desc.size
        num_values = 2
        nS = total_cells * num_values ** total_cells

        isd = np.zeros(desc.size).astype('float64')
        isd[0] = 1  # starting cell needs to be equal to 1, or for random start, assign all values to be equally divided amongst the whole

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def build_P(P):
            # init
            l = 0
            d_base = 2
            r = 1
            u_base = 0
            for p in P:
                # set done flag for finished state to include LDRU movement
                if 59 < p < 64:
                    done = True
                else:
                    done = False

                # update left
                left = (1, l, 0, done)

                # update down
                if p > 0 and p % 4 == 0:
                    d_base += 4
                    u_base += 4
                if p % 2 == 1:
                    d = d_base + 1
                    u = u_base + 1
                else:
                    d = d_base
                    u = u_base
                down = (1, d, 0, done)

                # update right
                right = (1, r, 0, done)

                # update up
                up = (1, u, 0, done)

                # write
                P[p][0].append(left)
                P[p][1].append(down)
                P[p][2].append(right)
                P[p][3].append(up)
                P[p][4].append(DIG[p])

                # update vars
                if p % 2 == 1:
                    l += 2
                    r += 2

            return P

        P = build_P(P)

        super(DiggerEnvDiscrete, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        # figure out which map to draw
        map = self.s // 4
        desc = MAPS[map]
        self.desc = desc = np.asarray(desc, dtype='c')
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]

        # map state to row/col to get robot position
        pos = self.s % 4
        if pos == 0:
            row = 0
            col = 0
        elif pos == 1:
            row = 0
            col = 1
        elif pos == 2:
            row = 1
            col = 0
        else:
            row = 1
            col = 1

        # color it
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up", "Dig"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc) + "\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
