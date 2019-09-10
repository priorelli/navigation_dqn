from __future__ import print_function
import numpy as np
import time
import rospy
import dqn_helpers as dhl
import nrp_helpers as nhl
from agent import Agent
import dqn_params as param
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach


def run_episode(episode, tau, q_primary, q_target):
    step, episode_done, rewards = 0, 0, []

    # the state contains the observation received from the environment
    # constructed by the sensory input of the robot
    pos, state = dhl.get_observation(nhl.raw_data)

    while not episode_done and step < param.steps:
        # form an observation vector based on the sensors
        prediction_q_actions = q_primary.feedforward(state)

        # choose action based on e-greedy policy
        if np.random.rand() < param.epsilon[episode]:
            action = np.random.randint(0, param.n_actions)
            print('Action (random):', nhl.get_action_name(action))
        else:
            action = int(np.argmax(prediction_q_actions))
           print('Action:', nhl.get_action_name(action))

        # the action goes to the transfer function and is executed
        action_done = 0
        rospy.set_param('action', action)
        rospy.set_param('action_done', action_done)

        # wait until action is done
        while action_done == 0:
            action_done = rospy.get_param('action_done')

        # the robot is now in a new state
        next_pos, next_state = dhl.get_observation(nhl.raw_data)
        print('Position:', int(next_pos[0]), int(next_pos[1]))
        print('Direction:', nhl.raw_data['direction'].data)
        print('Episode: ', episode + 1)
        print('Step:', step + 1)

        # check whether the agent received the reward
        reward, episode_done, reward_ind = nhl.get_reward(pos, next_pos)
        rewards.append(reward)
        print('Reward:', reward)
        print('-' * 10)

        # update weights
        loss = dhl.update_network_double(q_primary, q_target, state, next_state, reward)
        param.loss_of_episodes[episode].append(loss)
        print('Loss value:', loss)

        # copy weights and biases from target to primary
        if tau == param.max_tau - 1:
            q_target.weights = q_primary.weights.copy()
            q_target.biases = q_primary.biases.copy()

            tau = -1

        # set next state
        state = next_state
        pos = next_pos
        step += 1
        tau += 1

        if episode_done:
            print('Reward location found! Reward index:', reward_ind)
            print('-' * 10)

    # store metrics
    param.reward_of_episodes.append(sum(rewards))
    param.step_of_episodes.append(step)
    param.target_scores.append(episode_done)

    return tau


def main():
    # initialize agents
    q_primary = Agent(param.alpha, param.n_observations, param.n_actions, param.n_neurons)
    q_target = Agent(param.alpha, param.n_observations, param.n_actions, param.n_neurons)

    # if you want to run the experiment from an external program, use VC
    # this will allow you to use frontend interface from python
    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

    tau = 0  # update target network every max_tau steps
    for episode in range(param.episodes):
        count = 0
        while True:
            try:
                sim = vc.launch_experiment('template_husky_0')
                break
            except:
                count += 1
                print('Try:', count)
                time.sleep(2)

        # subscribe to topics published by ros
        nhl.perform_subscribers()

        # start the experiment
        sim.start()
        nhl.sync_params(episode)
        time.sleep(5)

        # inner-loop for running an episode
        tau = run_episode(episode, tau, q_primary, q_target)

        # stop experiment
        sim.stop()
        time.sleep(5)

        if episode % 100 == 0:
            # save metrics for postprocessing
            dhl.save_objects(q_primary, episode, 'double')

    print("Target score: ", sum(param.target_scores) / float(param.episodes))
    print('Reward visits:', np.sum(param.reward_visits))


if __name__ == '__main__':
    main()
