from __future__ import print_function
import numpy as np
import time
from env import Grid
import pickle
from dqn import Agent
import tensorflow as tf


def main():
    # set flags
    render = 0
    render_start = 0
    network = 'double'
    replay = 0

    # set model parameters
    episodes = 2000
    steps = 20
    max_tau = 10
    max_batch = 10

    gamma = 0.99
    alpha = 0.0005
    epsilon = np.linspace(0.5, 0.0, episodes)

    n_env = 11
    n_observations = 3
    n_actions = 3
    n_neurons = 128

    # create environment
    position = int((n_env - 1) / 2)
    env = Grid(n_env, (position, position), 0)

    total_steps = 0
    scores_manhattan = []
    scores_done = []
    losses = [[] for _ in range(episodes)]
    rewards_variation = []
    steps_variation = []

    # initialize networks
    q_primary = Agent(n_observations, n_actions, n_neurons)
    if network == 'double':
        q_target = Agent(n_observations, n_actions, n_neurons)
    optimizer = tf.optimizers.RMSprop(alpha)

    # start training
    tau = 0
    start = time.time()
    for episode in range(episodes):
        # set initial state
        state = env.reset()

        done = 0
        step = 0
        rewards = []
        batch = 0
        experience = []

        # start episode
        while not done and step < steps:
            if render_start or (render and episode > episodes - 2):
                env.render()

            # choose action and get reward
            action = np.random.randint(0, n_actions) if np.random.rand() < epsilon[episode] \
                         else int(np.argmax(q_primary.feedforward(state)))
            state_new, reward, done = env.step(action)
            rewards.append(reward)

            # update weights
            if replay:
                experience.append([state, state_new, reward])

                if batch == max_batch - 1 or batch == steps - 1:
                    with tf.GradientTape() as g:
                        prediction_q = tf.convert_to_tensor([q_primary.feedforward(i[0])[0] for i in experience])
                        target_q = tf.convert_to_tensor([q_primary.feedforward(i[1])[0] for i in experience]) \
                            if network == 'single' else tf.convert_to_tensor(
                            [q_target.feedforward(i[1])[0] for i in experience])
                        target = tf.cast(np.array([i[2] for i in experience])[np.newaxis, :].T + 
                                 gamma * np.max(target_q, axis=1)[np.newaxis, :].T, tf.float32)
                        loss = q_primary.get_loss(target, prediction_q)
                        losses[episode].append(loss.numpy())
                    
                        trainable_vars = list(q_primary.weights.values()) + list(q_primary.biases.values())
                        gradients = g.gradient(loss, trainable_vars)
                        optimizer.apply_gradients(zip(gradients, trainable_vars))
                     
                        experience = []

            else:
                with tf.GradientTape() as g:
                    prediction_q = q_primary.feedforward(state)
                    target_q = q_primary.feedforward(state_new) if network == 'single' \
                               else q_target.feedforward(state_new)
                    target = tf.cast(reward + gamma * np.max(target_q), tf.float32)
                    loss = q_primary.get_loss(target, prediction_q)
                    losses[episode].append(loss.numpy())
                
                    trainable_vars = list(q_primary.weights.values()) + list(q_primary.biases.values())
                    gradients = g.gradient(loss, trainable_vars)
                    optimizer.apply_gradients(zip(gradients, trainable_vars))

            if network == 'double' and tau == max_tau - 1:
                q_target.weights = q_primary.weights.copy()
                q_target.biases = q_primary.biases.copy()

            # set next state
            state = state_new
            step += 1
            tau = (tau + 1) % max_tau
            batch = (batch + 1) % max_batch

        # print reward
        scores_manhattan.append(env.dist / float(step) if done else 0.0)
        scores_done.append(done)
        rewards_variation.append(sum(rewards))
        steps_variation.append(step)
        if done:
            total_steps += step
        
        print('episode: %5d      reward: %6.2f      score: %6.2f      step: %2d      %s'
              % (episode + 1, sum(rewards), scores_manhattan[-1], step,
                 'Reward location is reached' if done else ''))

    # save objects
    pickle.dump(scores_manhattan, open('results/scores_manhattan_%s_%s.pkl' % (network, str(replay)), 'wb'))
    pickle.dump(scores_done, open('results/scores_done_%s_%s.pkl' % (network, str(replay)), 'wb'))
    pickle.dump(rewards_variation, open('results/rewards_variation_%s_%s.pkl' % (network, str(replay)), 'wb'))
    pickle.dump(steps_variation, open('results/steps_variation_%s_%s.pkl' % (network, str(replay)), 'wb'))
    pickle.dump(losses, open('results/losses_%s_%s.pkl' % (network, str(replay)), 'wb'))

    # print final score
    print('Time elapsed:', time.time() - start)
    print('Reward visits:', env.reward_visits)
    print('Score for reward visits:', sum(env.reward_visits.values()) / float(episodes))
    print('Score for steps:', sum(env.reward_visits.values()) / float(total_steps))


if __name__ == '__main__':
    main()
