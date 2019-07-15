from __future__ import print_function
import random
import numpy as np


# q-learning
class Qlearning:
    def __init__(self, gamma, alpha, n_env, n_actions):
        # set parameters
        self.gamma = gamma
        self.alpha = alpha
        self.n_env = n_env
        self.n_actions = n_actions

        # initialize table
        self.Q = np.zeros((self.n_env * self.n_env, self.n_actions))
        self.idxs = np.arange(self.n_env * self.n_env).reshape(self.n_env, self.n_env)

    def get_idx(self, s):
        return self.idxs[int(s[0]), int(s[1])]

    def choose_action(self, s, e):
        # epsilon greedy
        if random.random() < e:
            a = random.randint(0, self.n_actions - 1)
        else:
            a = int(np.argmax(self.Q[self.get_idx(s), :]))

        return a

    def update_table(self, s, s_n, a, r):
        td_error = r + self.gamma * np.max(self.Q[self.get_idx(s_n), :]) - self.Q[self.get_idx(s), a]
        self.Q[self.get_idx(s), a] += self.alpha * td_error
        return td_error
