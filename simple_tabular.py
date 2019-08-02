from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import time
from env import Grid
from qlearning import Qlearning
import tensorflow as tf


def main():
    render = 0
    render_start = 0
    plot = 1

    # set parameters
    episodes = 6000
    steps = 30

    gamma = 0.99
    alpha = 0.0005
    epsilon = np.linspace(0.5, 0.0, episodes)

    # create environment
    n_env = 9
    n_observation = 3
    n_actions = 3
    center = int((n_env - 1) / 2)
    env = Grid(n_env, (center, center), 0)

    start = time.time()
    
    scores = []
    losses = [[] for _ in range(episodes)]
    rewards_variation = []
    steps_variation = []

    qlearning = Qlearning(gamma, alpha, n_env, n_actions)

    # start training
    for episode in range(episodes):
        # set initial state
        state, dir_, distance = env.reset(f=False)

        done = 0
        step = 0
        rewards = []
        # start episode
        while not done and step < steps:
            if render_start or (render and episode > episodes - 2):
                env.render()

            action = qlearning.choose_action(state, dir_, epsilon[episode])

            # choose action and get reward
            state_new, dir_new, reward, done = env.step(action, f=False)
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

            # update weights
            loss = qlearning.update_table_qlearning(state, state_new, action, dir_, dir_new, reward)
            if step == 0:
                losses[episode].append(loss)
            else:
                losses[episode].append(losses[episode][-1] * .95 + loss * .05)
            # print('Loss:', loss)

            # set next state
            state = state_new
            dir_ = dir_new
            step += 1

        # print reward
        score = float(distance) / step if done else 0
        if episode == 0:
            scores.append(score)
            rewards_variation.append(sum(rewards))
            steps_variation.append(step)
        else:
            scores.append(scores[-1] * .95 + score * .05)
            rewards_variation.append(rewards_variation[-1] * .95 + sum(rewards) * .05)
            steps_variation.append(steps_variation[-1] * .95 + step * .05)

        print('episode: %5d      reward: %6.2f'
              '      score: %6.2f      step: %2d       %s'
              % (episode + 1, sum(rewards), scores[-1], step, 'Reward location is reached' if done else ''))

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

        rewardsplot.plot(np.arange(1, episodes + 1), rewards_variation)
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
