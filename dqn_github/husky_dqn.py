import tensorflow as tf
import numpy as np
import seaborn as sns
import utils as env
import scipy.io as sio

inits = tf.initializers.GlorotUniform()


class Agent(object):

    def __init__(self, observations, n_actions, h_units=256):
        self.weights = {
            'hidden1': tf.Variable(inits(shape=(observations, h_units))),
            'hidden2': tf.Variable(inits(shape=(h_units, h_units))),
            'hidden3': tf.Variable(inits(shape=(h_units, h_units))),
            'out': tf.Variable(inits(shape=(h_units, n_actions)))
        }

        self.biases = {
            'b1': tf.Variable(tf.random.normal([h_units])),
            'b2': tf.Variable(tf.random.normal([h_units])),
            'b3': tf.Variable(tf.random.normal([h_units])),
            'out': tf.Variable(tf.random.normal([n_actions]))
        }

    def feedforward(self, observations):
        x_inp = tf.cast(observations, tf.float32)
        layer1 = tf.nn.relu(tf.add(tf.matmul(x_inp, self.weights['hidden1']), self.biases['b1']))
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, self.weights['hidden2']), self.biases['b2']))
        layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, self.weights['hidden3']), self.biases['b3']))
        outputs = tf.add(tf.matmul(layer3, self.weights['out']), self.biases['out'])

        return outputs

    def get_loss(self, target, prediction):
        return tf.reduce_mean(tf.square(target - prediction))


def euclidean_dist(posx, posy, goalx, goaly):
    return np.sqrt((posx - goalx) ** 2 + (posx - goalx) ** 2)


def dists_to_rewards(rew_coordinates, posx, posy):
    dist_r1 = euclidean_dist(posx, posy, rew_coordinates[0][0], rew_coordinates[1][0])
    dist_r2 = euclidean_dist(posx, posy, rew_coordinates[0][1], rew_coordinates[1][1])
    dist_r3 = euclidean_dist(posx, posy, rew_coordinates[0][2], rew_coordinates[1][2])
    dist_r4 = euclidean_dist(posx, posy, rew_coordinates[0][3], rew_coordinates[1][3])

    return np.asarray([dist_r1, dist_r2, dist_r3, dist_r4])


def main():
    env_size = 17
    grd = sio.loadmat('grid_17.mat')
    grid = grd['grids']  # size(grid) 17x17

    grid = grid.astype(np.float32)
    grid_vect = np.asarray(grid.flatten())

    free_move, obs_val, rew_val = 0, -10, 10
    environment = env.generate_environment(env_size, obs_val, rew_val)

    environment_inds = np.arange(env_size ** 2).reshape(env_size, env_size)
    environmet_inds_list = np.arange(env_size ** 2)

    trajectory = env.generate_environment(env_size, obs_val, rew_val)

    action_dirs = ['Right', 'Forward', 'Left']
    action_inds = [[0, -1], [1, 0], [0, 1], [-1, 0]]

    nof_states, nof_actions = env_size ** 2, 4
    actions = np.arange(nof_actions)

    # trace reward locations
    left_top, left_bottom, right_top, right_bottom = 0, 0, 0, 0

    nof_episodes, nof_steps = range(3), range(1000)
    alpha, gamma, rand_act = 0.4, 0.03, 0.1

    rew_coordinates = np.where(environment == rew_val)

    # Q_mat = tf.Variable(tf.random.uniform([nof_states, nof_actions]))

    observation_len = 295
    model = Agent(observation_len, nof_actions, 128)

    # print(model.weights)

    # exit(0)

    optimizer = tf.optimizers.Adam(0.001)

    for i in nof_episodes:

        state = environmet_inds_list[int(len(environmet_inds_list) / 2)]
        trajectory = env.generate_environment(env_size, obs_val, rew_val)
        is_end, cum_reward = False, 0
        loss_trace, reward_trace = [], []

        for ii in nof_steps:

            xind, yind = np.where(environment_inds == state)

            # observations [agent_posx, agent_posy, [dist_rews], [grid_vect]
            dist_rews = dists_to_rewards(rew_coordinates, xind[0], yind[0])
            agent_posxy = np.asarray([xind[0].astype(np.float32), yind[0].astype(np.float32)])
            s_observation = np.concatenate((agent_posxy, dist_rews, grid_vect))
            s_observation = s_observation / np.mean(s_observation) * 10.0  # normalized version
            s_observation = np.reshape(s_observation, (1, s_observation.size))

            prediction_q = model.feedforward(s_observation)  # TODO: optimize it with the below

            if rand_act < np.random.rand():
                action = np.random.randint(nof_actions)
            else:
                action = tf.argmax(prediction_q, 1).numpy()[0]

            nx_ind, ny_ind = xind[0] + action_inds[action][0], yind[0] + action_inds[action][1]

            # check the boundary indices
            if nx_ind < 0 or nx_ind >= env_size:
                nx_ind = xind[0]
            if ny_ind < 0 or ny_ind >= env_size:
                ny_ind = yind[0]

            # print environment[nx_ind, ny_ind]
            if environment[nx_ind, ny_ind] == rew_val:
                # rew_location[nx_ind, ny_ind] += 1
                reward = 1
                is_end = True

            if environment[nx_ind, ny_ind] == obs_val:
                reward = -1.0
                # print "illegal action"
                nx_ind, ny_ind = xind[0], yind[0]

            if environment[nx_ind, ny_ind] == free_move:
                # print "Free move"
                reward = 0.0

            next_state_id = environment_inds[nx_ind, ny_ind]

            dist_rews = dists_to_rewards(rew_coordinates, nx_ind, ny_ind)
            agent_posxy = np.asarray([nx_ind.astype(np.float32), ny_ind.astype(np.float32)])
            ns_observation = np.concatenate((agent_posxy, dist_rews, grid_vect))
            ns_observation = ns_observation / np.mean(ns_observation) * 10.0  # normalized version
            ns_observation = np.reshape(ns_observation, (1, ns_observation.size))

            with tf.GradientTape() as g:
                prediction_q = model.feedforward(s_observation)  # returns Q(s, a_0),...., Q(s, a_3) # OK

                # exit(0)
                target_q = model.feedforward(ns_observation)
                target = tf.cast(reward + gamma * np.max(target_q), tf.float32)
                loss = model.get_loss(target, prediction_q)

            trainable_vars = list(model.weights.values()) + list(model.biases.values())
            gradients = g.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(gradients, trainable_vars))

            cum_reward += reward
            print("Step {} reward {} Loss {}".format(ii, cum_reward, loss))

            loss_trace.append(loss)
            reward_trace.append(cum_reward)

            state = next_state_id

            # rewards_trace.append(reward)
            trajectory[nx_ind, ny_ind] += 1
            sns.plt.figure("Agent trajectory")
            sns.heatmap(trajectory, linewidths=0.05, annot=False, cbar=False)

            sns.plt.figure("Cumulative reward")
            sns.plt.plot(reward_trace)
            sns.plt.xlim(0, len(nof_steps))

            sns.plt.figure("Loss")
            sns.plt.plot(loss_trace)
            sns.plt.xlim(0, len(nof_steps))

            # sns.plt.figure("foot steps")
            # sns.plt.scatter(nx_ind, ny_ind)
            # sns.plt.xlim(0, 20)
            # sns.plt.ylim(0, 20)

            sns.plt.show(block=False)
            sns.plt.pause(0.0001)

            if is_end:
                break


if __name__ == '__main__':
    main()
