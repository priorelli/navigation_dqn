import numpy as np
import seaborn as sns
import pickle


def running_average(x, window_size, mode='valid'):
    return np.convolve(x, np.ones(window_size) / window_size, mode=mode)


res_folder = 'dqn_main_results/'

t = 'single'
episode = 300

target_scores = pickle.load(open(res_folder + 'target_scores_%s_%d.pkl' % (t, episode), 'rb'))
reward_of_episodes = pickle.load(open(res_folder + 'reward_of_episodes_%s_%d.pkl' % (t, episode), 'rb'))
step_of_episodes = pickle.load(open(res_folder + 'step_of_episodes_%s_%d.pkl' % (t, episode), 'rb'))
loss_of_episodes = pickle.load(open(res_folder + 'loss_of_episodes_%s_%d.pkl' % (t, episode), 'rb'))
reward_visits = pickle.load(open(res_folder + 'reward_visits_%s_%d.pkl' % (t, episode), 'rb'))

w = episode / 5

fig1 = sns.plt.figure(1)
target_scores_plot = fig1.add_subplot(311)
reward_of_episodes_plot = fig1.add_subplot(312)
step_of_episodes_plot = fig1.add_subplot(313)

target_scores_avg = running_average(target_scores, w)
reward_of_episodes_avg = running_average(reward_of_episodes, w)
step_of_episodes_avg = running_average(step_of_episodes, w)

target_scores_plot.plot(np.arange(len(target_scores_avg)), target_scores_avg)
target_scores_plot.set_xlabel('Episode', fontweight='bold')
target_scores_plot.set_ylabel('Score', fontweight='bold')
target_scores_plot.grid(True)

reward_of_episodes_plot.plot(np.arange(len(reward_of_episodes_avg)), reward_of_episodes_avg)
reward_of_episodes_plot.set_xlabel('Episode', fontweight='bold')
reward_of_episodes_plot.set_ylabel('Reward', fontweight='bold')
reward_of_episodes_plot.grid(True)

step_of_episodes_plot.plot(np.arange(len(step_of_episodes_avg)), step_of_episodes_avg)
step_of_episodes_plot.set_xlabel('Episode', fontweight='bold')
step_of_episodes_plot.set_ylabel('Step', fontweight='bold')
step_of_episodes_plot.grid(True)

sns.plt.figure(2)
reward_visits_plot = sns.heatmap(reward_visits, annot=True, linewidths=0.2,
                                 cbar=False)
reward_visits_plot.set_title('Reward visits', fontweight='bold')

sns.plt.show()
