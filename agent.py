from __future__ import print_function
import tensorflow as tf
import numpy as np
import random


class Agent(object):

    def __init__(self, n_observations, n_actions, n_neurons, gamma, batch_size, tau, double=False):
        self.n_actions = n_actions
        self.n_neurons = n_neurons
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.double = double

        inits = tf.initializers.GlorotUniform()

        self.weights_primary = {
            'hidden1': tf.Variable(inits(shape=(n_observations, n_neurons))),
            'hidden2': tf.Variable(inits(shape=(n_neurons, n_neurons))),
            'out': tf.Variable(inits(shape=(n_neurons, n_actions)))
        }
        self.biases_primary = {
            'b1': tf.Variable(tf.random.normal([n_neurons])),
            'b2': tf.Variable(tf.random.normal([n_neurons])),
            'out': tf.Variable(tf.random.normal([n_actions]))
        }

        self.weights_target = {
            'hidden1': tf.Variable(inits(shape=(n_observations, n_neurons))),
            'hidden2': tf.Variable(inits(shape=(n_neurons, n_neurons))),
            'out': tf.Variable(inits(shape=(n_neurons, n_actions)))
        }
        self.biases_target = {
            'b1': tf.Variable(tf.random.normal([n_neurons])),
            'b2': tf.Variable(tf.random.normal([n_neurons])),
            'out': tf.Variable(tf.random.normal([n_actions]))
        }

        self.optimizer = tf.optimizers.Adam()

    def feedforward(self, observation):
        x_inp = tf.cast(observation, tf.float32)
        layer1 = tf.nn.relu(tf.add(tf.matmul(x_inp, self.weights_primary['hidden1']), self.biases_primary['b1']))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, self.weights_primary['hidden2']), self.biases_primary['b2']))
        output = tf.add(tf.matmul(layer2, self.weights_primary['out']), self.biases_primary['out'])

        return output

    def feedforward_batch(self, observations):
        outputs = [self.feedforward(observation.reshape(1, -1))[0] for observation in observations]

        return tf.convert_to_tensor(outputs)

    def feedforward_target(self, observations):
        outputs = []

        for observation in observations:
            x_inp = tf.cast(observation.reshape(1, -1), tf.float32)
            layer1 = tf.nn.relu(tf.add(tf.matmul(x_inp, self.weights_target['hidden1']), self.biases_target['b1']))
            layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, self.weights_target['hidden2']), self.biases_target['b2']))
            outputs.append(tf.add(tf.matmul(layer2, self.weights_target['out']), self.biases_target['out'])[0])

        return tf.convert_to_tensor(outputs)

    def choose_action(self, state, eps):
        if np.random.rand() < eps:
            return np.random.randint(0, self.n_actions)
        else:
            return np.argmax(self.feedforward(state.reshape(1, -1)))

    def train(self, memory):
        if len(memory.samples) < self.batch_size * 3:
            return 0
        batch = memory.sample(self.batch_size)
        states = np.array([val[0] for val in batch])
        actions = np.array([val[1] for val in batch])
        rewards = np.array([val[2] for val in batch])
        states_new = np.array([val[3] for val in batch])

        with tf.GradientTape() as g:
            pred_q = self.feedforward_batch(states)
            pred_q_new = self.feedforward_batch(states_new)

            batch_idxs = np.arange(self.batch_size)
            target_q = pred_q.numpy()

            if not self.double:
                updates = tf.cast(rewards + self.gamma * np.amax(pred_q_new.numpy(), axis=1), tf.float32)
            else:
                q_from_target = self.feedforward_target(states_new)
                updates = tf.cast(rewards + self.gamma * np.amax(q_from_target.numpy(), axis=1), tf.float32)

            target_q[batch_idxs, actions] = updates

            loss = tf.reduce_mean(tf.square(target_q - pred_q))
            trainable_vars_primary = list(self.weights_primary.values()) + list(self.biases_primary.values())
            gradients = g.gradient(loss, trainable_vars_primary)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars_primary))

        if self.double:
            trainable_vars_target = list(self.weights_target.values()) + list(self.biases_target.values())
            for t, p in zip(trainable_vars_target, trainable_vars_primary):
                t.assign(p * self.tau + t * (1 - self.tau))

        return loss.numpy()


class Memory:
    def __init__(self, max_memory):
        self.max_memory = max_memory
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.max_memory:
            self.samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self.samples):
            return random.sample(self.samples, len(self.samples))
        else:
            return random.sample(self.samples, no_samples)
