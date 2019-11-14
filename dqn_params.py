import numpy as np
import pickle


# model parameters
double = False
replay = True

n_env = 16  # size of the grid

episodes = 1000
steps = 32

batch_size = 6 if replay else 1
tau = 0.08

gamma = 0.9
lr = 0.01
max_eps = 0.8
min_eps = 0.01
decay = np.log(min_eps / (max_eps - min_eps)) / episodes
eps = max_eps

n_observations = 349  # 1207  # size of input to the network
n_actions = 3  # left, right and forward
n_neurons = 64  # number of units in hidden layer

# the first reward is top left and then clockwise direction
reward_poses = np.array([[1.5, 1.5], [1.5, 14.5], [14.5, 1.5], [14.5, 14.5]])
reward_target_found, reward_obstacle, reward_free = 10.0, -1.0, 0.0

# metrics
res_folder = 'results/'
str_ = 'double' if double else 'single'
str_ += '_replay' if replay else '_noreplay'

episode = 600

if episode:
    losses_variation = pickle.load(open('results/losses_variation_%s_%d.pkl' % (str_, episode), 'rb'))
    rewards_variation = pickle.load(open('results/rewards_variation_%s_%d.pkl' % (str_, episode), 'rb'))
    steps_variation = pickle.load(open('results/steps_variation_%s_%d.pkl' % (str_, episode), 'rb'))
    target_scores = pickle.load(open('results/target_scores_%s_%d.pkl' % (str_, episode), 'rb'))
    reward_visits = pickle.load(open('results/reward_visits_%s_%d.pkl' % (str_, episode), 'rb'))
else:
    losses_variation = [[] for _ in range(episodes)]
    rewards_variation = []
    steps_variation = []
    target_scores = []
    reward_visits = np.zeros((n_env, n_env), dtype=np.int32)
