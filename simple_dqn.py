from __future__ import print_function
import numpy as np
import time
from env import Grid
import pickle
from agent import Agent
import tensorflow as tf


def main():
    # set flags
    render = 0
    render_start = 0

    # set model parameters
    episodes = 1000
    steps = 36

    gamma = 0.99
    alpha = 0.0001
    epsilon = np.linspace(0.5, 0.05, episodes)

    n_env = 17
    n_observations = 3
    n_actions = 3
    n_neurons = 128

    # create environment
    position = int((n_env - 1) / 2)
    env = Grid(n_env, (position, position), 0)

    scores_manhattan = []
    scores_done = []
    rewards_variation = []
    steps_variation = []

    # initialize network
    dqn = Agent(alpha=alpha, n_observations=n_observations,
                n_actions=n_actions, h_units=n_neurons)

    # start training
    start = time.time()
    for episode in range(episodes):
        # set initial state
        state = env.reset()

        done = 0
        step = 0
        rewards = []

        # start episode
        while not done and step < steps:
            if render_start or (render and episode > episodes - 2):
                env.render()

            # choose action and get reward
            if np.random.rand() < epsilon[episode]:
                action = np.random.randint(0, n_actions)
            else:
                action = int(np.argmax(dqn.feedforward(state)))
            state_new, reward, done = env.step(action)
            rewards.append(reward)

            # update weights
            with tf.GradientTape() as g:
                prediction_q = dqn.feedforward(state)
                target_q = dqn.feedforward(state_new)
                target = tf.cast(reward + gamma * np.max(target_q), tf.float32)
                loss = dqn.get_loss(target, prediction_q)

                trainable_vars = list(dqn.weights.values()) + list(dqn.biases.values())
                gradients = g.gradient(loss, trainable_vars)
                dqn.optimizer.apply_gradients(zip(gradients, trainable_vars))

            # set next state
            state = state_new
            step += 1

        # print reward
        scores_manhattan.append(env.dist / float(step) if done else 0.0)
        scores_done.append(done)
        rewards_variation.append(sum(rewards))
        steps_variation.append(step)

        print('episode: %5d      reward: %6.2f      score: %6.2f      step: %2d      %s'
              % (episode + 1, sum(rewards), scores_manhattan[-1], step,
                 'Done' if done else ''))

    # save objects
    pickle.dump(scores_manhattan, open('results/scores_manhattan.pkl', 'wb'))
    pickle.dump(scores_done, open('results/scores_done.pkl', 'wb'))
    pickle.dump(rewards_variation, open('results/rewards_variation.pkl', 'wb'))
    pickle.dump(steps_variation, open('results/steps_variation.pkl', 'wb'))

    # print final score
    print('Time elapsed:', time.time() - start)
    print('Reward visits:', env.reward_visits)


if __name__ == '__main__':
    main()
