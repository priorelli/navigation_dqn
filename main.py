from __future__ import print_function
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
from grid_activations import generate_grids
from dqn.husky_dqn import Agent
import pickle
import tensorflow as tf
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach


def get_image(data):
    global camera

    cv_image = CvBridge().imgmsg_to_cv2(data, 'bgr8')[:170, :]
    cv_image = cv2.resize(cv_image, (32, 32))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    camera = cv_image.flatten().astype(np.float32)


def get_position(data):
    global position

    current_pose = data.pose[data.name.index('husky')]
    position = np.array([current_pose.position.x, current_pose.position.y])


rospy.Subscriber('husky/husky/camera', Image, get_image)
rospy.Subscriber('/gazebo/model_states', ModelStates, get_position)
env = '/home/spock/.opt/nrpStorage/template_husky_0_0_0_0/empty_world.sdf'


class Model:
    def __init__(self, episodes, steps, gamma, epsilon, alpha,
                 n_env, n_observation, n_actions):
        self.episodes = episodes
        self.steps = steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        self.n_env = n_env
        self.n_observation = n_observation
        self.n_actions = n_actions

        # initialize model
        self.model = Agent(self.n_observation, self.n_actions, 256)
        self.optimizer = tf.optimizers.Adam(0.001)
        self.grid_area = generate_grids(self.n_env)
        self.sim = None

        # create info and plots
        self.reward_visits = np.zeros((self.n_env, self.n_env), dtype=np.int32)
        self.scores = [0]
        self.losses = []

        self.initialize_plots(w='scores')

        # the first reward is top left and then clockwise direction
        self.reward_pos = np.array([[1.5, 1.5], [1.5, 14.5],
                                    [14.5, 1.5], [14.5, 14.5]])

    def initialize_plots(self, w=None):
        if w == 'scores':
            self.scoresmap = plt.figure(figsize=(5, 10))
            self.scoresmap1 = self.scoresmap.add_subplot(211)
            self.scoresmap2 = self.scoresmap.add_subplot(212)
        else:
            self.minimap = plt.figure(figsize=(5, 10))
            self.minimap1 = self.minimap.add_subplot(211)
            self.minimap2 = self.minimap.add_subplot(212)

            self.lossesmap = plt.figure(figsize=(5, 10))
            self.lossesmap1 = self.lossesmap.add_subplot(211)
            self.lossesmap2 = self.lossesmap.add_subplot(212)

    # concatenate and normalize inputs
    def build_input(self, pos, dir, cam):
        return np.concatenate((pos / 15.0, [dir / 3.0], cam / 255.0))[np.newaxis]

    def run_episode(self, i, pos, dir):
        global camera
        global position

        # draw minimap
        self.draw_map(i, pos, dir)

        step = 0
        done = 0
        idx = 0
        dist = sum(abs(self.reward_pos[0] - pos))
        state = self.build_input(pos, dir, camera)
        while not done and step < self.steps:
            # choose action using policy
            prediction_q = self.model.feedforward(state)
            if np.random.rand() < self.epsilon[i]:
                action = np.random.randint(0, self.n_actions)
            else:
                action = int(np.argmax(prediction_q))
            rospy.set_param('action', action)

            # execute action
            action_done = 0
            rospy.set_param('action_done', action_done)

            # wait until action is done
            while action_done == 0:
                action_done = rospy.get_param('action_done')
            time.sleep(.5)

            # get new state
            pos_new = position.copy()
            dir_new = rospy.get_param('direction')
            state_new = self.build_input(pos_new, dir_new, camera)

            # detect red
            red = rospy.get_param('red')
            print('Action:', 'move forward' if action == 0 else
                  'turn left' if action == 1 else 'turn right'
                  if action == 2 else '')
            print('Position:', pos_new[0], pos_new[1])
            print('Direction:', dir)
            print('Grid cell:', self.grid_area[int(pos_new[0]), int(pos_new[1])])

            # update plot
            self.draw_map(i, pos_new, dir)

            # get reward
            reward = 0
            if red:
                for r_idx, r_pos in enumerate(self.reward_pos):
                    if np.linalg.norm(pos_new- r_pos) <= 0.5:
                        reward = 5
                        done = 1
                        idx = r_idx
                        self.reward_visits[int(r_pos[0]), int(r_pos[1])] += 1
                        break
            elif np.linalg.norm(pos - pos_new) < .1:
                reward = -1

            # update weights
            with tf.GradientTape() as g:
                prediction_q = self.model.feedforward(state)
                target_q = self.model.feedforward(state_new)
                target = tf.cast(reward + self.gamma * np.max(target_q), tf.float32)
                loss = self.model.get_loss(target, prediction_q)

            trainable_vars = list(self.model.weights.values()) + list(self.model.biases.values())
            gradients = g.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # draw loss
            self.losses.append(loss.numpy())
            self.draw_loss(i, step)
            print('RMS loss:', loss.numpy())

            # set next state
            state = state_new
            pos = pos_new
            step += 1

        # draw scores
        score = dist / step if done else 0
        self.scores.append(self.scores[-1] * .9 + score * .1)
        self.draw_scores(i)

        if done:
            print('Done!')
            print('Reward index:', idx)
        print('Path length:', step, '\n')

    def run_training(self, vc):
        for i in range(self.episodes):
            rospy.set_param('i', i)

            # set initial parameters
            init_position = np.array([7.5, 7.5])
            init_direction = 0
            self.losses = []

            # launch experiment
            try:
                self.sim = vc.launch_experiment('template_husky_0_0_0_0')
            except:
                time.sleep(1)
            time.sleep(10)

            # start the experiment
            self.sim.start()

            # start episode
            self.run_episode(i, init_position, init_direction)

            # stop experiment
            self.sim.stop()
            time.sleep(10)

            # save objects
            pickle.dump(self.reward_visits, open('reward_visits.pkl', 'wb'))
            pickle.dump(self.losses, open('td_errors.pkl', 'wb'))
            pickle.dump(self.scores, open('scores.pkl', 'wb'))
            pickle.dump(self.model.weights, open('weights.pkl', 'wb'))
            pickle.dump(self.model.biases, open('biases.pkl', 'wb'))

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
        self.scoresmap2.set_ylabel('Score')
        self.scoresmap2.grid(True)

        # save plot
        self.scoresmap.savefig('plots/scores_%d.png' % i)

    def draw_loss(self, i, step):
        # plot losses
        if step > 0:
            self.lossesmap1.plot([step - 1, step], self.losses[step - 1: step + 1],
                                 linestyle='-', color='red')
        else:
            self.lossesmap1.plot(step, self.losses[step], linestyle='-', color='red')
        self.lossesmap1.set_xlabel('Step')
        self.lossesmap1.set_ylim([0, 5])
        self.lossesmap1.set_xlim([0, self.steps])
        self.lossesmap1.set_ylabel('Loss')
        self.lossesmap1.grid(True)

        # save plot
        self.lossesmap.savefig('plots/loss_%d.png' % i)


def main():
    # set parameters
    episodes = 100
    steps = 32

    gamma = 0.98
    epsilon = np.linspace(0.6, 0.0, episodes)
    alpha = 0.1

    n_env = 16
    n_observation = 1027
    n_actions = 3

    # start virtual coach
    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

    # run training
    model = Model(episodes=episodes, steps=steps, gamma=gamma, epsilon=epsilon,
                  alpha=alpha, n_env=n_env, n_observation=n_observation, n_actions=n_actions)
    model.run_training(vc)


if __name__ == '__main__':
    main()
