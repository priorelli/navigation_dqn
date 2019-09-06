import numpy as np
import seaborn as sns
import pickle


def running_average(x, window_size, mode='valid'):
    return np.convolve(x, np.ones(window_size) / window_size, mode=mode)


res_folder = 'dqn_main_results/'

t = '_single_0'

target_scores = pickle.load(open(res_folder + '%s.pkl' % ('target_scores' + t), 'rb'))
reward_of_episodes = pickle.load(open(res_folder + '%s.pkl' % ('reward_of_episodes' + t), 'rb'))
step_of_episodes = pickle.load(open(res_folder + '%s.pkl' % ('step_of_episodes' + t), 'rb'))
loss_of_episodes = pickle.load(open(res_folder + '%s.pkl' % ('loss_of_episodes' + t), 'rb'))
reward_visits = pickle.load(open(res_folder + '%s.pkl' % ('reward_visits' + t), 'rb'))

w = len(target_scores) / 5

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

# loss_plot = None
# loss_plot1 = None
# loss_plot2 = None
#
#
# def initialize_loss_plot():
#     loss_plot = plt.figure(figsize=(5, 10))
#     loss_plot1 = loss_plot.add_subplot(211)
#     loss_plot2 = loss_plot.add_subplot(212)
#
#
# def draw_loss(episode, step):
#     if step > 0:
#         loss_plot1.plot([step - 1, step], losses[step - 1: step + 1],
#                              linestyle='-', color='red')
#     else:
#         loss_plot1.plot(step, self.losses[step], linestyle='-', color='red')
#     loss_plot1.set_xlabel('Step')
#     loss_plot1.set_ylim([0, 5])
#     loss_plot1.set_xlim([0, self.steps])
#     loss_plot1.set_ylabel('Loss')
#     loss_plot1.grid(True)
#
#     # save plot
#     loss_plot.savefig('plots/loss_%d.png' % i)
#
#
#
# def draw_map(self, i, pos, dir_):
#     # plot robot position
#     markers = ['v', '>', '^', '<']
#     self.map_plot1.plot(pos[1], pos[0], marker=markers[dir_], markersize=3, color='red')
#     self.map_plot1.set_xlim([0, 16])
#     self.map_plot1.set_ylim([0, 16])
#     self.map_plot1.invert_yaxis()
#     ticks = np.arange(0, 16, 1)
#     self.map_plot1.set_xticks(ticks)
#     self.map_plot1.set_yticks(ticks)
#     self.map_plot1.grid(True)
#
#     # plot grid cells
#     self.map_plot2.imshow(self.grid_area, cmap='BuGn', interpolation='nearest')
#     self.map_plot2.invert_yaxis()
#     self.map_plot2.set_xticks([])
#     self.map_plot2.set_yticks([])
#     self.map_plot2.grid(False)
#
#     # save plot
#     self.map_plot.savefig('plots/map_%d.png' % i)
#
# def draw_scores(self, i):
#     # plot reward visits
#     self.scores_plot1.invert_yaxis()
#     ticks = np.arange(0, 16, 1)
#     self.scores_plot1.set_xticks(ticks)
#     self.scores_plot1.set_yticks(ticks)
#     self.scores_plot1.imshow(self.reward_visits, interpolation='none')
#
#     # plot scores
#     self.scores_plot2.set_xlim([0, self.episodes])
#     self.scores_plot2.set_ylim([0, 1])
#     self.scores_plot2.plot([i, i + 1], self.scores[i: i + 2], linestyle='-', color='red')
#     self.scores_plot2.set_ylabel('Score')
#     self.scores_plot2.grid(True)
#
#     # save plot
#     self.scores_plot.savefig('plots/scores_%d.png' % i)
