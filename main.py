from __future__ import print_function
import time
import logging
import rospy
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from grid_activations import generate_grids
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach


env = '/home/spock/.opt/nrpStorage/template_husky_0_0/empty_world.sdf'


class Model:
    def __init__(self, episodes, steps):
        self.episodes = episodes
        self.steps = steps

    def run_episode(self, i):
        t = 0
        done = 0

        while not done and t < self.steps:
            # choose random action
            action = random.randint(1, 3)
            rospy.set_param('action', action)
            print('action:', 'move forward' if action == 1 else
                  'turn left' if action == 2 else 'turn right'
                  if action == 3 else '')

            # execute action
            action_done = 0
            rospy.set_param('action_done', action_done)

            # wait until action is done
            while action_done == 0:
                action_done = rospy.get_param('action_done')
            pos = np.array(rospy.get_param('position'))
            dir = rospy.get_param('direction')
            red = rospy.get_param('red')
            print('position:', pos[0], pos[1])
            print('direction:', dir)
            print('grid cell:', self.grid_area[int(pos[0]), int(pos[1])], '\n')

            # update plot
            self.draw_plot(pos, dir)

            # get reward
            reward_pos = np.array([[1.5, 1.5], [14.5, 14.5], [1.5, 14.5], [14.5, 1.5]])
            if red and any(np.linalg.norm(pos - i) < 2 for i in reward_pos):
                reward = 5
                done = 1
                print('done!')

            t += 1
            time.sleep(.5)

        return

    def run_training(self, vc):
        # launch experiment
        try:
            self.sim = vc.launch_experiment('template_husky_0_0')
        except:
            time.sleep(1)
        time.sleep(10)

        for i in range(self.episodes):
            # create figure
            self.f = plt.figure(figsize=(5, 10))
            self.ax1 = self.f.add_subplot(211)
            self.ax2 = self.f.add_subplot(212)

            # set initial parameters
            init_position = np.array([7.5, 7.5])
            init_direction = 0
            self.grid_area = generate_grids(16)

            # draw plot
            self.draw_plot(init_position, init_direction)

            # start the experiment
            self.sim.start()

            # start episode
            self.run_episode(i)

            # reset simulation
            self.sim.reset('full')
            time.sleep(1)

    def draw_plot(self, pos, dir):
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
        self.ax2.imshow(self.grid_area, cmap='gray', interpolation='nearest')
        self.ax2.invert_yaxis()
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        self.ax2.grid(False)

        # save plot
        plt.savefig('plot.png')


def main():
    # set parameters
    episodes = 2000
    steps = 50

    # start virtual coach
    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

    # run training
    model = Model(episodes=episodes, steps=steps)
    model.run_training(vc)


if __name__ == '__main__':
    main()
