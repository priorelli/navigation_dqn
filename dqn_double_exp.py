from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
from env import Grid
import pickle
from dqn_github.husky_dqn import Agent
import tensorflow as tf


def main():
    render = 0
    render_start = 0
    plot = 0

    # set parameters
    episodes = 10000
    steps = 50
    max_tau = 30
    max_batch = 20

    gamma = 0.99
    alpha = 0.0005
    epsilon = np.linspace(0.5, 0.0, episodes)
    tau = 0

    # create environment
    n_env = 17
    n_observation = 3
    n_actions = 3
    center = int((n_env - 1) / 2)
    env = Grid(n_env, (center, center), 0)

    start = time.time()

    scores = []
    losses = [[] for _ in range(episodes)]
    rewards_variation = []
    steps_variation = []

    q_primary = Agent(n_observation, n_actions, 128)
    q_target = Agent(n_observation, n_actions, 128)
    optimizer = tf.optimizers.RMSprop(alpha)

    # start training
    for episode in range(episodes):
        # set initial state
        state, distance = env.reset()

        done = 0
        step = 0
        batch = 0
        experience = []
        rewards = []
        # start episode
        while not done and step < steps:
            if render_start or (render and episode > episodes - 2):
                env.render()

            if np.random.rand() < epsilon[episode]:
                action = np.random.randint(0, n_actions)
            else:
                action = int(np.argmax(q_primary.feedforward(state)))
            
            # choose action and get reward
            state_new, reward, done = env.step(action)
            rewards.append(reward)

            # print('Action:', 'move forward' if action == 0 else
            #       'turn left' if action == 1 else 'turn right'
            #       if action == 2 else '')
            # print('Position:', env.init[0], env.init[1])
            # print('Direction:', env.dir)
            # print('Reward:', reward)
            # if done:
            #     print('Done!')
            # print()

            experience.append([state, state_new, reward])

            if batch == max_batch - 1 or batch == steps - 1:
                # update weights
                with tf.GradientTape() as g:
                    predictions_q = tf.convert_to_tensor([q_primary.feedforward(i[0])[0] for i in experience])
                    targets_q = tf.convert_to_tensor([q_target.feedforward(i[1])[0] for i in experience])
                    targets = tf.cast(np.array([i[2] for i in experience])[np.newaxis, :].T + 
                    	gamma * np.max(targets_q, axis=1)[np.newaxis, :].T, tf.float32)
                    loss = q_primary.get_loss(targets, predictions_q)
                    losses[episode].append(loss)
                    # if step == 0:
                    #     losses[episode].append(loss)
                    # else:
                    #     losses[episode].append(losses[episode][-1] * .95 + loss * .05)
                    # print('RMS loss:', loss.numpy())

                trainable_vars = list(q_primary.weights.values()) + list(q_primary.biases.values())
                gradients = g.gradient(loss, trainable_vars)
                optimizer.apply_gradients(zip(gradients, trainable_vars))

                experience = []

                if tau == max_tau - 1:
                	q_target.weights = q_primary.weights.copy()
                	q_target.biases = q_primary.biases.copy()

            # set next state
            state = state_new
            step += 1
            batch = (batch + 1) % max_batch
            tau = (tau + 1) % max_tau

        # print reward
        score = float(distance) / step if done else 0
        scores.append(score)
        rewards_variation.append(sum(rewards))
        steps_variation.append(step)
        # if episode == 0:
        #     scores.append(score)
        #     rewards_variation.append(sum(rewards))
        #     steps_variation.append(step)
        # else:
        #     scores.append(scores[-1] * .95 + score * .05)
        #     rewards_variation.append(rewards_variation[-1] * .95 + sum(rewards) * .05)
        #     steps_variation.append(steps_variation[-1] * .95 + step * .05)

        print('episode: %5d      reward: %6.2f'
              '      score: %6.2f      step: %2d       %s'
              % (episode + 1, sum(rewards), scores[-1], step, 'Reward location is reached' if done else ''))

    # save objects
    pickle.dump(rewards_variation, open('results/rewards_variation_dqn_double_exp.pkl', 'wb'))
    pickle.dump(steps_variation, open('results/steps_variation_dqn_double_exp.pkl', 'wb'))
    pickle.dump(scores, open('results/scores_dqn_double_exp.pkl', 'wb'))
    pickle.dump(losses, open('results/losses_dqn_double_exp.pkl', 'wb'))

    # plot graph
    if plot:
        fig = plt.figure()
        scoreplot = fig.add_subplot(411)
        lossplot = fig.add_subplot(412)
        rewardsplot = fig.add_subplot(413)
        stepsplot = fig.add_subplot(414)
    
        scoreplot.plot(np.arange(1, episodes + 1), scores)
        scoreplot.set_xlabel('Episode')
        scoreplot.set_ylabel('Score')
        scoreplot.grid(True)
        
        for i in range(0, episodes, 500):
            lossplot.plot(np.arange(1, len(losses[i]) + 1), losses[i])
        lossplot.set_xlabel('Episode')
        lossplot.set_ylabel('Loss')
        lossplot.grid(True)

        rewardsplot.plot(np.arange(1, episodes + 1), rewards_variations)
        rewardsplot.set_xlabel('Episode')
        rewardsplot.set_ylabel('Rewards Variation')
        rewardsplot.grid(True)

        stepsplot.plot(np.arange(1, episodes + 1), steps_variation)
        stepsplot.set_xlabel('Episode')
        stepsplot.set_ylabel('Steps Variation')
        stepsplot.grid(True)

        plt.show()

    # print final score
    print('Time elapsed:', time.time() - start)
    print('Final score:', sum(env.reward_visits.values()) / float(episodes))
    print('Reward visits:', env.reward_visits)

if __name__ == '__main__':
    main()
