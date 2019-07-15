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

    def run_episode(self, i, state):
        step = 0
        done = 0

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
            self.draw_plot(i, state_new, dir)

            # get reward
            # first is top left and then clockwise direction
            reward_pos = np.array([[1.5, 1.5], [1.5, 14.5], [14.5, 1.5], [14.5, 14.5]])
            reward = 0
            if red:
                for r_idx, j in enumerate(reward_pos):
                    if np.linalg.norm(state_new - j) < 2 in reward_pos:
                        reward = 5
                        done = 1
                        self.reward_counters[int(j[0]), int(j[1])] += 1
                        print('Done!')
                        print('Reward index:', r_idx)
                        break
            elif np.linalg.norm(state - state_new) < .1:
                reward = -1

            # set next state
            state = state_new
            step += 1
            time.sleep(.5)

            # update table
            td = self.qlearning.update_table(state, state_new, action, reward)
            print('TD-error:', td)

        print('Path length:', step)

    def run_training(self, vc):

        for i in range(self.episodes):
            # create figure
            self.f = plt.figure(figsize=(5, 10))
            self.ax1 = self.f.add_subplot(211)
            self.ax2 = self.f.add_subplot(212)

            # set initial parameters
            init_position = np.array([7.5, 7.5])
            init_direction = 0
            self.grid_area = generate_grids(16)
            self.reward_counters = np.zeros((self.n_env, self.n_env), dtype=np.int32)

            # draw plot
            rospy.set_param('i', i)
            self.draw_plot(i, init_position, init_direction)

            # launch experiment
            try:
                self.sim = vc.launch_experiment('template_husky_0_0_0')
            except:
                time.sleep(1)
            time.sleep(10)

            # start the experiment
            self.sim.start()

            # start episode
            self.run_episode(i, init_position)

            # stop experiment
            self.sim.stop()
            time.sleep(10)

    def draw_plot(self, i, pos, dir):
        # plot robot position
        markers = ['v', '>', '^', '<']
        self.ax1.plot(pos[1], pos[0], marker=markers[dir], markersize=3, color='red')
        self.ax1.set_xlim([0, 16])
        self.ax1.set_ylim([0, 16])
        self.ax1.invert_yaxis()
        ticks = np.arange(0, 16, 1)
        self.ax1.set_xticks(ticks)
        self.ax1.set_yticks(ticks)
        self.ax1.grid(True)

        # plot grid cells
        self.ax2.imshow(self.grid_area, cmap='BuGn', interpolation='nearest')
        self.ax2.invert_yaxis()
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax2.grid(False)

        # save plot
        plt.savefig('plot_%d.png' % i)


def main():
    # set parameters
    episodes = 2000
    steps = 32

    gamma = .99
    epsilon = np.linspace(.6, .1, episodes)
    alpha = 1e-5

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
