from __future__ import print_function
import numpy as np
import time
from env import Grid
import pickle
from qlearning import Qlearning


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
    n_actions = 3

    # create environment
    position = int((n_env - 1) / 2)
    env = Grid(n_env, (position, position), 0)

    scores_manhattan = []
    scores_done = []
    rewards_variation = []
    steps_variation = []

    # initialize network
    qlearning = Qlearning(gamma, alpha, n_env, n_actions)

    # start training
    start = time.time()
    for episode in range(episodes):
        # set initial state
        state, dir_, = env.reset(f=False)

        done = 0
        step = 0
        rewards = []

        # start episode
        while not done and step < steps:
            if render_start or (render and episode > episodes - 2):
                env.render()

            # choose action and get reward
            action = qlearning.choose_action(state, dir_, epsilon[episode])
            state_new, dir_new, reward, done = env.step(action, f=False)
            rewards.append(reward)

            # update weights
            _ = qlearning.update_table_qlearning(state, state_new, action, dir_, dir_new, reward)

            # set next state
            state = state_new
            dir_ = dir_new
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
