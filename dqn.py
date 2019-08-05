import tensorflow as tf
import utils as env

inits = tf.initializers.GlorotUniform()


class Agent(object):

    def __init__(self, observations, n_actions, h_units=256):
        self.weights = {
            'hidden1': tf.Variable(inits(shape=(observations, h_units))),
            'hidden2': tf.Variable(inits(shape=(h_units, h_units))),
            'out': tf.Variable(inits(shape=(h_units, n_actions)))
        }

        self.biases = {
            'b1': tf.Variable(tf.random.normal([h_units])),
            'b2': tf.Variable(tf.random.normal([h_units])),
            'out': tf.Variable(tf.random.normal([n_actions]))
        }

    def feedforward(self, observations):
        x_inp = tf.cast(observations, tf.float32)
        layer1 = tf.nn.relu(tf.add(tf.matmul(x_inp, self.weights['hidden1']), self.biases['b1']))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, self.weights['hidden2']), self.biases['b2']))
        outputs = tf.add(tf.matmul(layer2, self.weights['out']), self.biases['out'])

        return outputs

    def get_loss(self, target, prediction):
        return tf.reduce_mean(tf.square(target - prediction))
