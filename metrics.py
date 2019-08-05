import numpy as np
import matplotlib.pyplot as plt
import pickle


def integrate(x):
	a = [x[0]]
	for i in range(1, len(x)):
		a.append(a[i - 1] * 0.99 + x[i] * 0.01)
	return a


def running_average(x, window_size, mode='valid'):
	return np.convolve(x, np.ones(window_size) / window_size, mode=mode)


episodes = 20000
w = 5000

t = 'double_1'

scores_manhattan_name = 'scores_manhattan_' + t
scores_done_name = 'scores_done_' + t
rewards_variation_name = 'rewards_variation_' + t
steps_variation_name = 'steps_variation_' + t

scores_manhattan = pickle.load(open('results/%s.pkl' % scores_manhattan_name, 'rb'))
scores_done = pickle.load(open('results/%s.pkl' % scores_done_name, 'rb'))
rewards_variation = pickle.load(open('results/%s.pkl' % rewards_variation_name, 'rb'))
steps_variation = pickle.load(open('results/%s.pkl' % steps_variation_name, 'rb'))

fig = plt.figure()
scores_manhattan_plot = fig.add_subplot(411)
scores_done_plot = fig.add_subplot(412)
rewards_variation_plot = fig.add_subplot(413)
steps_variation_plot = fig.add_subplot(414)

scores_manhattan_avg = running_average(scores_manhattan, w)
scores_done_avg = running_average(scores_done, w)
rewards_variation_avg = running_average(rewards_variation, w)
steps_variation_avg = running_average(steps_variation, w)

scores_manhattan_plot.plot(np.arange(1, len(scores_manhattan_avg) + 1), scores_manhattan_avg)
scores_manhattan_plot.set_xlabel('Episode')
scores_manhattan_plot.set_ylabel('Score')
scores_manhattan_plot.grid(True)

scores_done_plot.plot(np.arange(1, len(scores_done_avg) + 1), scores_done_avg)
scores_done_plot.set_xlabel('Episode')
scores_done_plot.set_ylabel('Score')
scores_done_plot.grid(True)

rewards_variation_plot.plot(np.arange(1, len(rewards_variation_avg) + 1), rewards_variation_avg)
rewards_variation_plot.set_xlabel('Episode')
rewards_variation_plot.set_ylabel('Reward')
rewards_variation_plot.grid(True)

steps_variation_plot.plot(np.arange(1, len(steps_variation_avg) + 1), steps_variation_avg)
steps_variation_plot.set_xlabel('Episode')
steps_variation_plot.set_ylabel('Step')
steps_variation_plot.grid(True)

plt.show()
