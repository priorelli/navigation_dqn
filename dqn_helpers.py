import time
import cv2
import numpy as np
import dqn_params as param
import tensorflow as tf
from cv_bridge import CvBridge
import pickle


def tic():
    """ Log starting time to run specific code block."""

    global _start_time
    _start_time = time.time()


def toc():
    """ Print logged time in hour : min : sec format. """

    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)

    return '%f hours: %f mins: %f secs' % (t_hour, t_min, t_sec)


def process_camera(camera):
    # crop the ROI for removing husky front part
    camera = CvBridge().imgmsg_to_cv2(camera, 'bgr8')[:170, :]
    camera = cv2.resize(camera, (32, 32))
    camera = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
    camera = camera.flatten().astype(np.float32)

    return camera


def process_laser(laser):
    laser = np.asarray(laser.ranges)
    laser[laser == np.inf] = 0.0
    laser = laser[np.arange(0, len(laser), 2)]

    return laser


def process_position(position):
    position = position.pose[position.name.index('husky')]
    position = np.asarray([position.position.x, position.position.y])

    return position


def construct_input(position, direction, camera, laser):
    return np.concatenate((position, [direction], camera, laser))


def normalize_input(input_):
    min_input, max_input = np.min(input_), np.max(input_)
    normalized_input = (input_ - min_input) / (max_input - min_input)

    return normalized_input[np.newaxis, :]


def get_observation(raw_data):
    position = process_position(raw_data['position'])
    direction = float(raw_data['direction'].data)
    camera = process_camera(raw_data['camera'])
    laser = process_laser(raw_data['laser'])

    input_ = construct_input(position, direction, camera, laser)

    return position, normalize_input(input_)


def update_network_single(q_primary, state, next_state, reward):
    with tf.GradientTape() as g:
        prediction_q = q_primary.feedforward(state)
        target_q = q_primary.feedforward(next_state)
        target = tf.cast(reward + param.gamma * np.max(target_q), tf.float32)
        loss = q_primary.get_loss(target, prediction_q)

        trainable_vars = list(q_primary.weights.values()) + list(q_primary.biases.values())
        gradients = g.gradient(loss, trainable_vars)
        q_primary.optimizer.apply_gradients(zip(gradients, trainable_vars))

    return loss.numpy()


def update_network_double(q_primary, q_target, state, next_state, reward):
    with tf.GradientTape() as g:
        prediction_q = q_primary.feedforward(state)
        target_q = q_target.feedforward(next_state)
        target = tf.cast(reward + param.gamma * np.max(target_q), tf.float32)
        loss = q_primary.get_loss(target, prediction_q)

        trainable_vars = list(q_primary.weights.values()) + list(q_primary.biases.values())
        gradients = g.gradient(loss, trainable_vars)
        q_primary.optimizer.apply_gradients(zip(gradients, trainable_vars))

    return loss.numpy()


def update_network_single_replay(q_primary, experience):
    with tf.GradientTape() as g:
        prediction_q = tf.convert_to_tensor([q_primary.feedforward(i[0])[0] for i in experience])
        target_q = tf.convert_to_tensor([q_primary.feedforward(i[1])[0] for i in experience])
        target = tf.cast(np.asarray([i[2] for i in experience])[np.newaxis, :].T +
                         param.gamma * np.max(target_q, axis=1)[np.newaxis, :].T, tf.float32)
        loss = q_primary.get_loss(target, prediction_q)

        trainable_vars = list(q_primary.weights.values()) + list(q_primary.biases.values())
        gradients = g.gradient(loss, trainable_vars)
        q_primary.optimizer.apply_gradients(zip(gradients, trainable_vars))

    return loss.numpy()


def update_network_double_replay(q_primary, q_target, experience):
    with tf.GradientTape() as g:
        prediction_q = tf.convert_to_tensor([q_primary.feedforward(i[0])[0] for i in experience])
        target_q = tf.convert_to_tensor([q_target.feedforward(i[1])[0] for i in experience])
        target = tf.cast(np.asarray([i[2] for i in experience])[np.newaxis, :].T +
                         param.gamma * np.max(target_q, axis=1)[np.newaxis, :].T, tf.float32)
        loss = q_primary.get_loss(target, prediction_q)

        trainable_vars = list(q_primary.weights.values()) + list(q_primary.biases.values())
        gradients = g.gradient(loss, trainable_vars)
        q_primary.optimizer.apply_gradients(zip(gradients, trainable_vars))

    return loss.numpy()


def save_objects(network, episode, type_):
    pickle.dump(param.loss_of_episodes, open(param.res_folder + 'loss_of_episodes_%s_%d.pkl'
                                             % (type_, episode), 'wb'))
    pickle.dump(param.reward_of_episodes, open(param.res_folder + 'reward_of_episodes_%s_%d.pkl'
                                               % (type_, episode), 'wb'))
    pickle.dump(param.step_of_episodes, open(param.res_folder + 'step_of_episodes_%s_%d.pkl'
                                             % (type_, episode), 'wb'))
    pickle.dump(param.target_scores, open(param.res_folder + 'target_scores_%s_%d.pkl'
                                          % (type_, episode), 'wb'))
    pickle.dump(param.reward_visits, open(param.res_folder + 'reward_visits_%s_%d.pkl'
                                          % (type_, episode), 'wb'))

    pickle.dump(network.weights, open(param.res_folder + 'weights_%s_%d.pkl'
                                      % (type_, episode), 'wb'))
    pickle.dump(network.biases, open(param.res_folder + 'biases_%s_%d.pkl'
                                     % (type_, episode), 'wb'))
