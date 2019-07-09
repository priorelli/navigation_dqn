from __future__ import print_function
import time
import logging
import rospy
import random
import numpy as np
import matplotlib.pyplot as plt
import grid_activations
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach


env = '/home/spock/.opt/nrpStorage/template_husky_0/empty_world.sdf'


class Model:
    def __init__(self, episodes, steps):
        self.episodes = episodes
        self.steps = steps
        # self.actions = [1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2,
        #                 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2]

    def run_episode(self, i, pos, dir, grid):
        t = 0
        done = 0

        while not done and t < self.steps:
            # action = self.actions[t]
            action = random.randint(1, 3)
            rospy.set_param('action', action)
            print('action:', 'move forward' if action == 1 else
                  'turn left' if action == 2 else 'turn right'
                  if action == 3 else '')

            # execute action
            action_done = 0
            rospy.set_param('action_done', action_done)
            while action_done == 0:
                action_done = rospy.get_param('action_done')
            pos = np.array(rospy.get_param('position'))
            dir = np.array(rospy.get_param('direction'))
            red = np.array(rospy.get_param('red'))
            print('position:', pos[0], pos[1])
            print('head direction:', dir)
            print('grid cell:', grid[int(pos[0]) - 1, int(pos[1]) - 1])
            print()

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
            self.sim = vc.launch_experiment('template_husky_0')
        except:
            time.sleep(1)
        time.sleep(10)

        for i in range(self.episodes):
            init_position = np.array([7.5, 7.5])
            init_direction = 0.
            init_grid = grid_activations.generate_grids(14)
            # init_state = np.concatenate([init_grid.flatten(), [init_orientation]])

            # start the experiment
            self.sim.start()

            # start episode
            self.run_episode(i, init_position, init_direction, init_grid)

            self.sim.reset('robot_pose')
            time.sleep(1)


def main():
    # set parameters
    episodes = 2000
    steps = 20

    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

    model = Model(episodes=episodes, steps=steps)
    model.run_training(vc)


if __name__ == '__main__':
    main()
