import numpy as np
import seaborn as sns
import pickle


def running_average(x, window_size, mode='valid'):
	return np.convolve(x, np.ones(window_size) / window_size, mode=mode)


scores_manhattan = pickle.load(open('results/scores_manhattan.pkl', 'rb'))
scores_done = pickle.load(open('results/scores_done.pkl', 'rb'))
rewards_variation = pickle.load(open('results/rewards_variation.pkl', 'rb'))
steps_variation = pickle.load(open('results/steps_variation.pkl', 'rb'))

w = int(len(scores_manhattan) / 5)


fig = sns.plt.figure(1)
scores_manhattan_plot = fig.add_subplot(411)
scores_done_plot = fig.add_subplot(412)
rewards_variation_plot = fig.add_subplot(413)
steps_variation_plot = fig.add_subplot(414)

scores_manhattan_avg = running_average(scores_manhattan, w)
scores_done_avg = running_average(scores_done, w)
rewards_variation_avg = running_average(rewards_variation, w)
steps_variation_avg = running_average(steps_variation, w)

scores_manhattan_plot.plot(np.arange(len(scores_manhattan_avg)), scores_manhattan_avg)
scores_manhattan_plot.set_xlabel('Episode', fontweight='bold')
scores_manhattan_plot.set_ylabel('Score', fontweight='bold')
scores_manhattan_plot.grid(True)

scores_done_plot.plot(np.arange(len(scores_done_avg)), scores_done_avg)
scores_done_plot.set_xlabel('Episode', fontweight='bold')
scores_done_plot.set_ylabel('Score', fontweight='bold')
scores_done_plot.grid(True)

rewards_variation_plot.plot(np.arange(len(rewards_variation_avg)), rewards_variation_avg)
rewards_variation_plot.set_xlabel('Episode', fontweight='bold')
rewards_variation_plot.set_ylabel('Reward', fontweight='bold')
rewards_variation_plot.grid(True)

steps_variation_plot.plot(np.arange(len(steps_variation_avg)), steps_variation_avg)
steps_variation_plot.set_xlabel('Episode', fontweight='bold')
steps_variation_plot.set_ylabel('Step', fontweight='bold')
steps_variation_plot.grid(True)

sns.plt.show()
