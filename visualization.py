import numpy as np
import dqn_params as param
import seaborn as sns
import pickle


def running_average(x, window_size, mode='valid'):
    return np.convolve(x, np.ones(window_size) / window_size, mode=mode)


t = 'single_replay'
episode = 800

target_scores = pickle.load(open(param.res_folder + 'target_scores_%s_%d.pkl' % (t, episode), 'rb'))
reward_of_episodes = pickle.load(open(param.res_folder + 'rewards_variation_%s_%d.pkl' % (t, episode), 'rb'))
step_of_episodes = pickle.load(open(param.res_folder + 'steps_variation_%s_%d.pkl' % (t, episode), 'rb'))
loss_of_episodes = pickle.load(open(param.res_folder + 'losses_variation_%s_%d.pkl' % (t, episode), 'rb'))
reward_visits = pickle.load(open(param.res_folder + 'reward_visits_%s_%d.pkl' % (t, episode), 'rb'))

w = episode / 5

fig1 = sns.plt.figure(1)
target_scores_plot = fig1.add_subplot(311)
reward_of_episodes_plot = fig1.add_subplot(312)
step_of_episodes_plot = fig1.add_subplot(313)

target_scores_avg = running_average(target_scores, w)
reward_of_episodes_avg = running_average(reward_of_episodes, w)
step_of_episodes_avg = running_average(step_of_episodes, w)

s = 20

target_scores_plot.plot(np.arange(len(target_scores_avg)), target_scores_avg, linewidth=3.0)
target_scores_plot.set_xlabel('Episode', fontweight='bold', fontsize=s)
target_scores_plot.set_ylabel('Score', fontweight='bold', fontsize=s)
target_scores_plot.grid(True)

reward_of_episodes_plot.plot(np.arange(len(reward_of_episodes_avg)), reward_of_episodes_avg, linewidth=3.0)
reward_of_episodes_plot.set_xlabel('Episode', fontweight='bold', fontsize=s)
reward_of_episodes_plot.set_ylabel('Reward', fontweight='bold', fontsize=s)
reward_of_episodes_plot.grid(True)

step_of_episodes_plot.plot(np.arange(len(step_of_episodes_avg)), step_of_episodes_avg, linewidth=3.0)
step_of_episodes_plot.set_xlabel('Episode', fontweight='bold', fontsize=s)
step_of_episodes_plot.set_ylabel('Step', fontweight='bold', fontsize=s)
step_of_episodes_plot.grid(True)

sns.plt.figure(2)
reward_visits_plot = sns.heatmap(reward_visits, annot=True, linewidths=0.2,
                                 cbar=False)
reward_visits_plot.set_title('Reward visits', fontweight='bold')

sns.plt.show()
