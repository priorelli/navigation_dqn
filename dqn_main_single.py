from __future__ import print_function
import numpy as np
import time
import rospy
import dqn_helpers as dhl
import nrp_helpers as nhl
from agent import Agent
import dqn_params as param
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach


def run_episode(episode, q_primary):
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
        rospy.set_param('action', action)

        # wait until action is done
        while action != -1:
            time.sleep(0.2)
            try:
                action = rospy.get_param('action')
            except KeyError:
                pass

        # the robot is now in a new state
        next_pos, next_state = dhl.get_observation(nhl.raw_data)
        print('Position: %.1f %.1f' % (next_pos[0], next_pos[1]))
        print('Direction:', nhl.raw_data['direction'].data)
        print('Episode: ', episode + 1)
        print('Step:', step + 1)

        # check whether the agent received the reward
        reward, episode_done, reward_ind = nhl.get_reward(pos, next_pos)
        rewards.append(reward)
        print('Reward:', reward)
        print('-' * 10)

        # update weights
        loss = dhl.update_network_single(q_primary, state, next_state, reward)
        param.loss_of_episodes[episode].append(loss)
        print('Loss value:', loss)

        # set next state
        state = next_state
        pos = next_pos
        step += 1

        if episode_done:
            print('Reward location found! Reward index:', reward_ind)
            print('-' * 10)

    # store metrics
    param.reward_of_episodes.append(sum(rewards))
    param.step_of_episodes.append(step)
    param.target_scores.append(episode_done)


def main():
    # initialize agents
    q_primary = Agent(param.alpha, param.n_observations, param.n_actions, param.n_neurons)

    # if you want to run the experiment from an external program, use VC
    # this will allow you to use frontend interface from python
    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

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
        nhl.sync_params(episode + 1)
        sim.start()
        time.sleep(5)

        # inner-loop for running an episode
        run_episode(episode, q_primary)

        # stop experiment
        sim.stop()
        time.sleep(5)

        # save metrics and network for postprocessing
        if (episode + 1) % 100 == 0:
            dhl.save_objects(q_primary, episode + 1, 'single')

    print('Target score:', sum(param.target_scores) / float(param.episodes))
    print('Reward visits:', np.sum(param.reward_visits))


if __name__ == '__main__':
    main()
