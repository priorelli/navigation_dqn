from __future__ import print_function
import numpy as np
import pickle
import time
from agent import Agent, Memory
from env import Grid


def main():
    # set flags
    render = 0
    render_start = 0

    # set model parameters
    episodes = 500
    steps = 34
    batch_size = 12
    tau = 0.08

    gamma = 0.98
    max_eps = 0.8
    min_eps = 0.01
    decay = np.log(min_eps / (max_eps - min_eps)) / episodes

    n_env = 17
    n_observations = 3
    n_actions = 3
    n_neurons = 64

    # create environment
    env = Grid(n_env)

    # initialize network
    agent = Agent(n_observations, n_actions, n_neurons, gamma, batch_size, tau, double=True)
    memory = Memory(50000)

    scores_manhattan = []
    scores_done = []
    rewards_variation = []
    steps_variation = []

    # start training
    start = time.time()
    eps = max_eps
    for episode in range(episodes):
        # set initial state
        state = env.reset()
        done, step, tot_reward = 0, 0, 0

        while not done and step < steps:
            if render_start or (render and episode > episodes - 2):
                env.render()

            # choose action and get reward
            action = agent.choose_action(state, eps)
            state_new, reward, done = env.step(action)
            tot_reward += reward

            # store in memory
            memory.add_sample((state, action, reward, state_new))

            # train network
            loss = agent.train(memory)

            # set next state
            state = state_new
            step += 1

        # decay epsilon function
        eps = (max_eps - min_eps) * np.exp(decay * episode)

        # store info
        scores_manhattan.append(env.dist / float(step) if done else 0.0)
        scores_done.append(done)
        rewards_variation.append(tot_reward)
        steps_variation.append(step)

        print('episode: %5d      reward: %6.2f      score: %6.2f      step: %2d      %s'
              % (episode + 1, tot_reward, scores_manhattan[-1], step, 'Done' if done else ''))

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
