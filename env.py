import numpy as np
import matplotlib.pyplot as plt


# navigation task
class Grid:
    def __init__(self, dim, robot, dir_):
        self.dim = dim
        self.robot = robot
        self.dir = dir_

        self.grid = np.array([])
        self.init = ()
        self.goals = []

        self.reward_visits = {(1, 1): 0, (1, self.dim - 2): 0,
                              (self.dim - 2, 1): 0, (self.dim - 2, self.dim - 2): 0}

        self.dist = 0

    # reset environment
    def reset(self, f=True):
        # reset grid
        self.grid = [[1 for _ in range(self.dim)] for _
                     in range(2)]
        for _ in range(self.dim - 2):
            self.grid.insert(1, [1] + [0 for _ in
                             range(self.dim - 2)] + [1])
        self.grid = np.array(self.grid)

        # reset initial and final positions
        self.init = self.robot

        self.goals = [(1, 1), (1, self.dim - 2),
                      (self.dim - 2, 1), (self.dim - 2, self.dim - 2)]

        self.grid[self.init] = 2
        for goal in self.goals:
            self.grid[goal] = 3

        self.dist = np.sum(np.abs(
            np.array(self.init) - np.array(self.goals[0])))

        if f:
            # normalized = np.concatenate([self.init, [self.dir]])
            # min_, max_ = np.min(normalized), np.max(normalized)
            # normalized = (normalized - min_) / float(max_ - min_)
            return np.concatenate([np.array(self.init) / float(self.dim),
                                  [self.dir / 3.0]])[np.newaxis, :]
            # return normalized[np.newaxis, :]

        else:
            return self.init, self.dir

    # make one step
    def step(self, action, f=True):
        bounce = 0

        x, y = self.init

        self.dir = (self.dir + 1) % 4 if action == 1 else \
            (self.dir - 1) % 4 if action == 2 else self.dir

        inc = (0, 1) if self.dir == 0 else (
            1, 0) if self.dir == 1 else \
            (0, -1) if self.dir == 2 else (-1, 0)

        if self.grid[x + inc[0], y + inc[1]] != 1:
            self.grid[x, y] = 0
            self.grid[x + inc[0], y + inc[1]] = 2
            self.init = (x + inc[0], y + inc[1])
            for goal in self.goals:
                self.grid[goal] = 3
        else:
            bounce = 1

        # compute reward
        reward = 10.0 if self.init in self.goals else 0.0
        reward -= 1.0 if bounce else 0.0

        done = 1 if self.init in self.goals else 0

        if done:
            self.reward_visits[self.init] += 1

        if f:
            # normalized = np.concatenate([self.init, [self.dir]])
            # min_, max_ = np.min(normalized), np.max(normalized)
            # normalized = (normalized - min_) / float(max_ - min_)
            return np.concatenate([np.array(self.init) / float(self.dim),
                                   [self.dir / 3.0]])[np.newaxis, :], reward, done
            # return normalized[np.newaxis, :], reward, done
        else:
            return self.init, self.dir, reward, done

    # render world
    def render(self):
        plt.imshow(self.grid, interpolation='none')
        ticks = np.arange(0, self.dim, 1)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.grid(False)
        plt.show()
