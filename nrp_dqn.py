from __future__ import print_function
import time
import rospy
import numpy as np
import matplotlib.pyplot as plt
from grid_activations import generate_grids
from dqn import Agent
import pickle
import tensorflow as tf
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelStates
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach


# get data from camera
def get_image(data):
    global camera

    cv_image = CvBridge().imgmsg_to_cv2(data, 'bgr8')[:170, :]
    cv_image = cv2.resize(cv_image, (32, 32))
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    camera = cv_image.flatten().astype(np.float32)


# get robot position
def get_position(data):
    global position

    current_pose = data.pose[data.name.index('husky')]
    position = np.array([current_pose.position.x, current_pose.position.y])


rospy.Subscriber('husky/husky/camera', Image, get_image)
rospy.Subscriber('/gazebo/model_states', ModelStates, get_position)
env = '/home/spock/.opt/nrpStorage/template_husky_0_0_0_0/empty_world.sdf'


class Model:
    def __init__(self, episodes, steps, max_tau, max_batch, gamma, epsilon, alpha,
                 n_env, n_observations, n_actions, n_neurons, network, replay):
        # set model parameters
        self.episodes = episodes
        self.steps = steps
        self.max_tau = max_tau
        self.max_batch = max_batch

        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha

        self.n_env = n_env
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_neurons = n_neurons

        self.network = network
        self.replay = replay

        self.tau = 0

        # initialize networks
        self.q_primary = Agent(self.n_observations, self.n_actions, self.n_neurons)
        if self.network == 'double':
            self.q_target = Agent(self.n_observations, self.n_actions, self.n_neurons)
        self.optimizer = tf.optimizers.RMSprop(self.alpha)
        self.grid_area = generate_grids(self.n_env)
        self.sim = None

        # create info and plots
        self.reward_visits = np.zeros((self.n_env, self.n_env), dtype=np.int32)
        self.scores = []
        self.losses = []

        # initialize scores plot
        self.initialize_plots(w='scores')

        # the first reward is top left and then clockwise direction
        self.reward_pos = np.array([[1.5, 1.5], [1.5, 14.5],
                                    [14.5, 1.5], [14.5, 14.5]])

    def initialize_plots(self, w=None):
        if w == 'scores':
            self.scores_plot = plt.figure(figsize=(5, 10))
            self.scores_plot1 = self.scores_plot.add_subplot(211)
            self.scores_plot2 = self.scores_plot.add_subplot(212)
        else:
            self.map_plot = plt.figure(figsize=(5, 10))
            self.map_plot1 = self.map_plot.add_subplot(211)
            self.map_plot2 = self.map_plot.add_subplot(212)

            self.loss_plot = plt.figure(figsize=(5, 10))
            self.loss_plot1 = self.loss_plot.add_subplot(211)
            self.loss_plot2 = self.loss_plot.add_subplot(212)

    # concatenate and normalize inputs
    def build_input(self, pos, dir_, cam):
        return np.concatenate((pos / 15.0, [dir_ / 3.0], cam / 255.0))[np.newaxis]

    def run_episode(self, i, pos, dir_):
        global camera
        global position

        # initialize plots and draw map
        self.initialize_plots()
        self.draw_map(i, pos, dir_)

        step = 0
        done = 0
        idx = 0
        batch = 0
        rewards = []
        experience = []
        dist = sum(abs(self.reward_pos[0] - pos))

        state = self.build_input(pos, dir_, camera)
        while not done and step < self.steps:
            # choose action using policy
            prediction_q = self.q_primary.feedforward(state)
            action = np.random.randint(0, self.n_actions) if np.random.rand() < self.epsilon[i] \
                         else action = int(np.argmax(prediction_q))

            rospy.set_param('action', action)
            print('Action:', 'Move forward' if action == 0 else
                  'Turn left' if action == 1 else 'Turn right'
                  if action == 2 else '')

            # execute action
            action_done = 0
            rospy.set_param('action_done', action_done)

            # wait until action is done
            while action_done == 0:
                action_done = rospy.get_param('action_done')
            time.sleep(0.5)

            # get new state
            pos_new = position.copy()
            dir_new = rospy.get_param('direction')
            state_new = self.build_input(pos_new, dir_new, camera)

            # detect red
            red = rospy.get_param('red')

            print('Position:', pos_new[0], pos_new[1])
            print('Direction:', dir_new)
            print('Grid cell:', self.grid_area[int(pos_new[0]), int(pos_new[1])])

            # update map
            self.draw_map(i, pos_new, dir_new)

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
            rewards.append(reward)

            # update weights
            if self.replay:
                experience.append([state, state_new, reward])

                if batch == self.max_batch - 1 or batch == self.steps - 1:
                    with tf.GradientTape() as g:
                        prediction_q = tf.convert_to_tensor([self.q_primary.feedforward(i[0])[0] for i in experience])
                        target_q = tf.convert_to_tensor([self.q_primary.feedforward(i[1])[0] for i in experience]) \
                            if self.network == 'single' else tf.convert_to_tensor(
                            [self.q_target.feedforward(i[1])[0] for i in experience])
                        target = tf.cast(np.array([i[2] for i in experience])[np.newaxis, :].T +
                                         self.gamma * np.max(target_q, axis=1)[np.newaxis, :].T, tf.float32)
                        loss = self.q_primary.get_loss(target, prediction_q)
                        self.losses.append(loss)

                        trainable_vars = list(self.q_primary.weights.values()) + list(self.q_primary.biases.values())
                        gradients = g.gradient(loss, trainable_vars)
                        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

                        experience = []

            else:
                with tf.GradientTape() as g:
                    prediction_q = self.q_primary.feedforward(state)
                    target_q = self.q_primary.feedforward(state_new)
                    target = tf.cast(reward + self.gamma * np.max(target_q), tf.float32)
                    loss = self.q_primary.get_loss(target, prediction_q)
                    self.losses.append(loss.numpy())

                trainable_vars = list(self.q_primary.weights.values()) + list(self.q_primary.biases.values())
                gradients = g.gradient(loss, trainable_vars)
                self.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # draw loss
            self.draw_loss(i, step)
            print('RMS loss:', loss.numpy(), '\n')

            # set next state
            state = state_new
            pos = pos_new
            step += 1

        # draw scores
        score = dist / float(step) if done else 0.0
        self.scores.append(self.scores[-1] * 0.99 + score * 0.01)
        self.draw_scores(i)

        if done:
            print('Done with reward index:', idx)
        print('Path length:', step, '\n')

    def run_training(self, vc):
        for i in range(self.episodes):
            self.losses = []
            rospy.set_param('i', i)

            # set initial parameters
            init_position = np.array([7.5, 7.5])
            init_direction = 0

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
        pickle.dump(self.q_primary.weights, open('weights.pkl', 'wb'))
        pickle.dump(self.q_primary.biases, open('biases.pkl', 'wb'))

    def draw_map(self, i, pos, dir_):
        # plot robot position
        markers = ['v', '>', '^', '<']
        self.map_plot1.plot(pos[1], pos[0], marker=markers[dir_], markersize=3, color='red')
        self.map_plot1.set_xlim([0, 16])
        self.map_plot1.set_ylim([0, 16])
        self.map_plot1.invert_yaxis()
        ticks = np.arange(0, 16, 1)
        self.map_plot1.set_xticks(ticks)
        self.map_plot1.set_yticks(ticks)
        self.map_plot1.grid(True)

        # plot grid cells
        self.map_plot2.imshow(self.grid_area, cmap='BuGn', interpolation='nearest')
        self.map_plot2.invert_yaxis()
        self.map_plot2.set_xticks([])
        self.map_plot2.set_yticks([])
        self.map_plot2.grid(False)

        # save plot
        self.map_plot.savefig('plots/map_%d.png' % i)

    def draw_scores(self, i):
        # plot reward visits
        self.scores_plot1.invert_yaxis()
        ticks = np.arange(0, 16, 1)
        self.scores_plot1.set_xticks(ticks)
        self.scores_plot1.set_yticks(ticks)
        self.scores_plot1.imshow(self.reward_visits, interpolation='none')

        # plot scores
        self.scores_plot2.set_xlim([0, self.episodes])
        self.scores_plot2.set_ylim([0, 1])
        self.scores_plot2.plot([i, i + 1], self.scores[i: i + 2], linestyle='-', color='red')
        self.scores_plot2.set_ylabel('Score')
        self.scores_plot2.grid(True)

        # save plot
        self.scores_plot.savefig('plots/scores_%d.png' % i)

    def draw_loss(self, i, step):
        # plot losses
        if step > 0:
            self.loss_plot1.plot([step - 1, step], self.losses[step - 1: step + 1],
                                 linestyle='-', color='red')
        else:
            self.loss_plot1.plot(step, self.losses[step], linestyle='-', color='red')
        self.loss_plot1.set_xlabel('Step')
        self.loss_plot1.set_ylim([0, 5])
        self.loss_plot1.set_xlim([0, self.steps])
        self.loss_plot1.set_ylabel('Loss')
        self.loss_plot1.grid(True)

        # save plot
        self.loss_plot.savefig('plots/loss_%d.png' % i)


def main():
    # set flags
    network = 'single'
    replay = 0

    # set model parameters
    episodes = 10000
    steps = 50
    max_tau = 30
    max_batch = 20

    gamma = 0.99
    epsilon = np.linspace(0.5, 0.0, episodes)
    alpha = 0.001

    n_env = 16
    n_observations = 1027
    n_actions = 3
    n_neurons = 256

    # start virtual coach
    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

    # run training
    model = Model(episodes=episodes, steps=steps, max_tau=max_tau, max_batch=max_batch,
                  gamma=gamma, epsilon=epsilon, alpha=alpha, n_env=n_env,
                  n_observations=n_observations, n_actions=n_actions, n_neurons=n_neurons,
                  network=network, replay=replay)
    model.run_training(vc)


if __name__ == '__main__':
    main()
