import numpy as np
import matplotlib.pyplot as plt


# navigation task
class Grid:
    def __init__(self, dim, init_pos, init_dir):
        self.dim = dim
        self.init_pos = init_pos
        self.init_dir = init_dir

        self.pos = (0, 0)
        self.dir_ = 0
        self.grid = np.array([])

        self.goals = [(1, 1), (1, self.dim - 2),
                      (self.dim - 2, 1), (self.dim - 2, self.dim - 2)]

        self.reward_visits = {(1, 1): 0, (1, self.dim - 2): 0,
                              (self.dim - 2, 1): 0, (self.dim - 2, self.dim - 2): 0}

        self.dist = np.sum(np.abs(np.array(self.init_pos) - np.array(self.goals[0])))

    # reset environment
    def reset(self, f=True):
        # reset grid
        self.grid = [[1 for _ in range(self.dim)] for _ in range(2)]
        for _ in range(self.dim - 2):
            self.grid.insert(1, [1] + [0 for _ in range(self.dim - 2)] + [1])
        self.grid = np.array(self.grid)

        # reset initial and final positions
        self.pos = self.init_pos
        self.dir_ = self.init_dir

        self.grid[self.pos] = 2
        for goal in self.goals:
            self.grid[goal] = 3

        if f:
            return np.concatenate([np.array(self.pos) / float(self.dim),
                                  [self.dir_ / 3.0]])[np.newaxis, :]
        else:
            return self.pos, self.dir_

    # make one step
    def step(self, action, f=True):
        bounce = 0

        x, y = self.pos

        self.dir_ = (self.dir_ + 1) % 4 if action == 1 else \
            (self.dir_ - 1) % 4 if action == 2 else self.dir_

        inc = (0, 1) if self.dir_ == 0 else (
            1, 0) if self.dir_ == 1 else \
            (0, -1) if self.dir_ == 2 else (-1, 0)

        if self.grid[x + inc[0], y + inc[1]] != 1:
            self.grid[x, y] = 0
            self.grid[x + inc[0], y + inc[1]] = 2
            self.pos = (x + inc[0], y + inc[1])
            for goal in self.goals:
                self.grid[goal] = 3
        else:
            bounce = 1

        # compute reward
        reward = 10.0 if self.pos in self.goals else 0.0
        reward -= 2.0 if bounce else 0.0

        done = 1 if self.pos in self.goals else 0

        if done:
            self.reward_visits[self.pos] += 1

        if f:
            return np.concatenate([np.array(self.pos) / float(self.dim),
                                   [self.dir_ / 3.0]])[np.newaxis, :], reward, done
        else:
            return self.pos, self.dir_, reward, done

    # render world
    def render(self):
        plt.imshow(self.grid, interpolation='none')
        ticks = np.arange(0, self.dim, 1)
        plt.xticks(ticks)
        plt.yticks(ticks)
        plt.grid(False)
        plt.show()
