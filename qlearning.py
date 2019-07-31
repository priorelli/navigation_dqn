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
        self.Q = np.zeros((self.n_env * self.n_env * 4, self.n_actions))
        self.idxs = np.arange(self.n_env * self.n_env * 4).reshape(self.n_env, self.n_env, 4)

    def get_idx(self, s, d):
        return self.idxs[int(s[0]), int(s[1]), d]

    def choose_action(self, s, d, e):
        # epsilon greedy
        if random.random() < e:
            a = random.randint(0, self.n_actions - 1)
        else:
            a = int(np.argmax(self.Q[self.get_idx(s, d), :]))

        return a

    def update_table_qlearning(self, s, s_n, a, d, d_n, r):
        td_error = r + self.gamma * np.max(self.Q[self.get_idx(s_n, d_n), :]) - self.Q[self.get_idx(s, d), a]
        self.Q[self.get_idx(s, d), a] += self.alpha * td_error
        return td_error

    def update_table_sarsa(self, s, s_n, a, a_n, d, d_n, r):
        td_error = r + self.gamma * self.Q[self.get_idx(s_n, d_n), a_n] - self.Q[self.get_idx(s, d), a]
        self.Q[self.get_idx(s, d), a] += self.alpha * td_error
        return td_error
