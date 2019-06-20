from __future__ import print_function
import time
import logging
import rospy
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import grid_activations
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach


env = '/home/spock/.opt/nrpStorage/template_husky_0/empty_world.sdf'


class Model:
    def __init__(self, episodes, steps):
        self.episodes = episodes
        self.steps = steps
        self.actions = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2]

    def run_episode(self, i):
        t = 0

        while t < self.steps:
            # choose action
            # action = random.randint(1, 3)
            action = self.actions[t]
            rospy.set_param('action', action)
            print('action:', 'move forward' if action == 1 else
                  'turn left' if action == 2 else 'turn right'
                  if action == 3 else '')

            # execute action
            action_done = 0
            rospy.set_param('action_done', action_done)
            while action_done == 0:
                action_done = rospy.get_param('action_done')
            # position = np.array(rospy.get_param('position'))
            # orientation = np.array(rospy.get_param('orientation'))
            # print('position:', position)
            # print('orientation:', orientation)

            t += 1
            time.sleep(.5)

        return

    def run_training(self, vc):
        # launch experiment
        for i in range(self.episodes):
            try:
                self.sim = vc.launch_experiment('template_husky_0')
            except:
                time.sleep(1)
            time.sleep(20)

            # start the experiment
            self.sim.start()

            # start episode
            self.run_episode(i)

            self.sim.stop()
            time.sleep(20)


def main():
    # set parameters
    episodes = 1
    steps = 28

    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

    model = Model(episodes=episodes, steps=steps)
    model.run_training(vc)


if __name__ == '__main__':
    main()
