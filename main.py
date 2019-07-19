from __future__ import print_function
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
from qlearning import Qlearning
from grid_activations import generate_grids
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach


env = '/home/spock/.opt/nrpStorage/template_husky_0_0_0/empty_world.sdf'


class Model:
    def __init__(self, episodes, steps, gamma, epsilon, alpha,
                 n_env, n_actions):
        self.episodes = episodes
        self.steps = steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.n_env = n_env
        self.n_actions = n_actions

        self.qlearning = Qlearning(gamma=self.gamma, alpha=self.alpha,
                                   n_env=self.n_env, n_actions=self.n_actions)
        self.reward_visits = np.zeros((self.n_env, self.n_env), dtype=np.int32)
        self.scores = [0]

        # first is top left and then clockwise direction
        self.reward_pos = np.array([[1.5, 1.5], [1.5, 14.5], [14.5, 1.5], [14.5, 14.5]])

    def run_episode(self, i, state):
        step = 0
        done = 0

        dist = sum(abs(self.reward_pos[0] - state))

        # action = self.qlearning.choose_action(state, self.epsilon[i])
        while not done and step < self.steps:
            # choose action using policy
            action = self.qlearning.choose_action(state, self.epsilon[i])
            rospy.set_param('action', action)
            print('Action:', 'move forward' if action == 0 else
                  'turn left' if action == 1 else 'turn right'
                  if action == 2 else '')

            # execute action
            action_done = 0
            rospy.set_param('action_done', action_done)

            # wait until action is done
            while action_done == 0:
                action_done = rospy.get_param('action_done')
            state_new = np.array(rospy.get_param('position'))
            dir = rospy.get_param('direction')
            red = rospy.get_param('red')
            print('Position:', state_new[0], state_new[1])
            print('Direction:', dir)
            print('Grid cell:', self.grid_area[
                int(state_new[0]), int(state_new[1])], '\n')

            # update plot
            self.draw_map(i, state_new, dir)

            # get reward
            reward = 0
            if red:
                print('red')
                for r_idx, j in enumerate(self.reward_pos):
                    if np.linalg.norm(state_new - j) < 0.6:
                        reward = 5
                        done = 1
                        self.reward_visits[int(j[0]), int(j[1])] += 1
                        print('Reward index:', r_idx)
                        break
            elif np.linalg.norm(state - state_new) < .1:
                reward = -1

            # action_new = self.qlearning.choose_action(state, self.epsilon[i])

            # update table
            td = self.qlearning.update_table_qlearning(state, state_new, action, reward)
            # td = self.qlearning.update_table_sarsa(state, state_new, action, action_new, reward)
            self.tderrors.append(td)
            print('TD-error:', td)
            self.draw_tderror(i, step)

            # set next state
            state = state_new
            # action = action_new
            step += 1
            time.sleep(.5)

        score = dist / step if done else 0
        self.scores.append(self.scores[-1] * .8 + score * .2)
        print('Path length:', step)

    def run_training(self, vc):

        for i in range(self.episodes):
            # create figures
            self.minimap = plt.figure(figsize=(5, 10))
            self.minimap1 = self.minimap.add_subplot(211)
            self.minimap2 = self.minimap.add_subplot(212)

            self.scoresmap = plt.figure(figsize=(5, 10))
            self.scoresmap1 = self.scoresmap.add_subplot(211)
            self.scoresmap2 = self.scoresmap.add_subplot(212)

            self.tderrormap = plt.figure()
            self.tderrormap1 = self.tderrormap.add_subplot(111)

            # set initial parameters
            init_position = np.array([11.5, 11.5])
            init_direction = 0
            self.grid_area = generate_grids(16)
            self.tderrors = []

            # draw minimap
            rospy.set_param('i', i)
            self.draw_map(i, init_position, init_direction)

            # launch experiment
            try:
                self.sim = vc.launch_experiment('template_husky_0_0_0')
            except:
                time.sleep(1)
            time.sleep(5)

            # start the experiment
            self.sim.start()

            # start episode
            self.run_episode(i, init_position)

            # stop experiment
            self.sim.stop()
            time.sleep(5)

            # draw scores map
            self.draw_scores(i)

    def draw_map(self, i, pos, dir):
        # plot robot position
        markers = ['v', '>', '^', '<']
        self.minimap1.plot(pos[1], pos[0], marker=markers[dir], markersize=3, color='red')
        self.minimap1.set_xlim([0, 16])
        self.minimap1.set_ylim([0, 16])
        self.minimap1.invert_yaxis()
        ticks = np.arange(0, 16, 1)
        self.minimap1.set_xticks(ticks)
        self.minimap1.set_yticks(ticks)
        self.minimap1.grid(True)

        # plot grid cells
        self.minimap2.imshow(self.grid_area, cmap='BuGn', interpolation='nearest')
        self.minimap2.invert_yaxis()
        self.minimap2.set_xticks([])
        self.minimap2.set_yticks([])
        self.minimap2.grid(False)

        # save plot
        self.minimap.savefig('plots/plot_%d.png' % i)

    def draw_scores(self, i):
        # plot reward visits
        self.scoresmap1.invert_yaxis()
        ticks = np.arange(0, 16, 1)
        self.scoresmap1.set_xticks(ticks)
        self.scoresmap1.set_yticks(ticks)
        self.scoresmap1.imshow(self.reward_visits, interpolation='none')

        # plot scores
        self.scoresmap2.set_xlim([0, self.episodes])
        self.scoresmap2.set_ylim([0, 1])
        self.scoresmap2.plot([i, i + 1], self.scores[i: i + 2], linestyle='-', color='red')
        self.scoresmap2.set_xlabel('Episode')
        self.scoresmap2.set_ylabel('Score')
        self.scoresmap2.grid(True)

        # save plot
        self.scoresmap.savefig('plots/scores_%d.png' % i)

    def draw_tderror(self, i, step):
        try:
            self.tderrormap1.plot([step - 1, step], self.tderrors[step - 1: step + 1],
                                  linestyle='-', color='red')
        except:
            self.tderrormap1.plot(step, self.tderrors[step], linestyle='-', color='red')
        self.tderrormap1.set_xlabel('Step')
        self.tderrormap1.set_ylim([0, 100])
        self.tderrormap1.set_ylabel('TD-error')
        self.tderrormap1.grid(True)

        self.tderrormap.savefig('plots/tderror_%d.png' % i)


def main():
    # set parameters
    episodes = 2000
    steps = 32

    gamma = .98
    epsilon = np.linspace(.6, .1, episodes)
    alpha = .1

    n_env = 16
    n_actions = 3

    # start virtual coach
    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

    # run training
    model = Model(episodes=episodes, steps=steps, gamma=gamma, epsilon=epsilon,
                  alpha=alpha, n_env=n_env, n_actions=n_actions)
    model.run_training(vc)


if __name__ == '__main__':
    main()
