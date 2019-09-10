# this file contains experiment parameters for simple dqn
import numpy as np


# model parameters
n_env = 16  # size of the grid

episodes = 1000
steps = 36

max_tau = 20
max_batch = 10

gamma = 0.9
epsilon = np.linspace(0.5, 0.05, episodes)
alpha = 0.0001   # learning rate

n_observations = 1207  # size of input to the network
n_actions = 3  # left, right and forward
n_neurons = 128  # number of units in hidden layer

# the first reward is top left and then clockwise direction
reward_poses = np.array([[1.5, 1.5], [1.5, 14.5], [14.5, 1.5], [14.5, 14.5]])
reward_target_found, reward_obstacle, reward_free = 10.0, -2.0, 0.0

# metrics
res_folder = 'results/'

loss_of_episodes = [[] for _ in range(episodes)]
reward_of_episodes = []
step_of_episodes = []
target_scores = []  # 1 if the agent reached the goal, otherwise 0
reward_visits = np.zeros((n_env, n_env), dtype=np.int32)
