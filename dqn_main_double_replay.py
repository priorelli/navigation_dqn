from __future__ import print_function
import numpy as np
import dqn_helpers as dhl
import nrp_helpers as nhl
import time
import rospy
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach
from agent import Agent
import dqn_params as param
import pickle


def run_episode(episode, tau, q_primary, q_target):
    step, episode_done, rewards = 0, 0, []
    batch, experience = 0, []

    # state contains the observation received from the environment
    # constructed by the sensory input of the robot
    pos, state = dhl.get_observation(nhl.raw_data)

    while not episode_done and step < param.steps:
        # form an observation vector based on the sensors
        prediction_q_actions = q_primary.feedforward(state)

        # choose action based on e-greedy policy
        if np.random.rand() < param.epsilon[episode]:
            action = np.random.randint(0, param.n_actions)
            print('-----Random action-----')
        else:
            action = int(np.argmax(prediction_q_actions))
        print('Action:', nhl.get_action_name(action))

        # the action goes to the transfer function
        rospy.set_param('action', action)

        # execute action
        action_done = 0
        rospy.set_param('action_done', action_done)

        # wait until action is done
        while action_done == 0:
            action_done = rospy.get_param('action_done')

        # the robot will take an action, now it is in a next state
        next_pos, next_state = dhl.get_observation(nhl.raw_data)
        print('Position:', int(next_pos[0]), int(next_pos[1]))
        print('Direction:', nhl.raw_data['direction'].data)
        print('Episode: ', episode)
        print('Remaining episodes:', param.episodes - episode)
        print('Remaining steps:', param.steps - step)
        print('-' * 10)

        # check whether the agent received the reward
        reward, episode_done, reward_ind = nhl.get_reward(pos, next_pos)
        print('Reward:', reward)
        rewards.append(reward)
        if episode_done:
            print('Reward location found! Reward index:', reward_ind)

        # add state to experience replay and update weights
        experience.append((state, next_state, reward))

        if batch == param.max_batch - 1 or step == param.steps - 1 or episode_done:
            loss = dhl.update_network_double_replay(q_primary, q_target, experience)
            param.loss_of_episodes[episode].append(loss)
            print('Loss value:', loss)

            experience = []
            batch = -1

        if tau == param.max_tau - 1:
            q_target.weights = q_primary.weights.copy()
            q_target.biases = q_primary.biases.copy()

            tau = -1

        # set next state
        state = next_state
        pos = next_pos
        step += 1
        tau += 1
        batch += 1

    param.reward_of_episodes.append(sum(rewards))
    param.step_of_episodes.append(step)
    param.target_scores.append(episode_done)

    return tau


def main():
    res_folder = 'dqn_main_results/'

    # initialize agents
    q_primary = Agent(param.alpha, param.n_observations, param.n_actions, param.n_neurons)
    q_target = Agent(param.alpha, param.n_observations, param.n_actions, param.n_neurons)

    # if you want to run the experiment from an external program, use VC
    # this will allow you to use frontend interface from python
    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

    # subscribe to topics published by ros
    nhl.perform_subscribers()

    tau = 0  # update the second network after max_tau steps
    for episode in range(param.episodes):
        sim = vc.launch_experiment('template_husky_0')

        # start the experiment
        sim.start()
        nhl.sync_params()
        time.sleep(5)

        # time-stamp for the video streaming
        rospy.set_param('i', episode)

        # inner-loop for running an episode
        tau = run_episode(episode, tau, q_primary, q_target)

        # stop experiment
        sim.stop()
        time.sleep(5)

        if episode % 200 == 0:
            # save metrics for postprocessing
            pickle.dump(param.loss_of_episodes, open(res_folder + 'loss_of_episodes_double_replay_%d.pkl'
                        % episode, 'wb'))
            pickle.dump(param.reward_of_episodes, open(res_folder + 'reward_of_episodes_double_replay_%d.pkl'
                        % episode, 'wb'))
            pickle.dump(param.step_of_episodes, open(res_folder + 'step_of_episodes_double_replay_%d.pkl'
                        % episode, 'wb'))
            pickle.dump(param.target_scores, open(res_folder + 'target_scores_double_replay_%d.pkl'
                        % episode, 'wb'))
            pickle.dump(param.reward_visits, open(res_folder + 'reward_visits_double_replay_%d.pkl'
                        % episode, 'wb'))

            # save the network
            pickle.dump(q_primary.weights, open(res_folder + 'weights_double_replay_%d.pkl'
                        % episode, 'wb'))
            pickle.dump(q_primary.biases, open(res_folder + 'biases_double_replay_%d.pkl'
                        % episode, 'wb'))

    print("Target score: ", sum(param.target_scores) / float(param.episodes))
    print('Reward visits:', np.sum(param.reward_visits))


if __name__ == '__main__':
    main()
