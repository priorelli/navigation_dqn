from __future__ import print_function
import numpy as np
import time
import rospy
import pickle
import dqn_helpers as dhl
import nrp_helpers as nhl
import dqn_params as param
from agent import Agent, Memory
from std_msgs.msg import Int32
from hbp_nrp_virtual_coach.virtual_coach import VirtualCoach


action_topic = rospy.Publisher('action_topic', Int32, queue_size=1)
folder = '~/.opt/nrpStorage/template_husky_0/'


def run_episode(start_pos, episode, agent, memory):
    step, episode_done, tot_reward = 0, 0, 0

    # the state contains the observation received from the environment
    # constructed by the sensory input of the robot
    pos, state = dhl.get_observation(nhl.raw_data)

    if np.linalg.norm(pos - start_pos) > 0.2:
        print('Fail')
        return False

    while not episode_done and step < param.steps:
        # choose action based on e-greedy policy
        action = agent.choose_action(state, param.eps)
        # print('Action:', nhl.get_action_name(action))

        # the action goes to the transfer function and is executed
        action_topic.publish(Int32(action))

        # wait until action is done
        while not nhl.raw_data['action_done'].data:
            time.sleep(0.1)
        action_topic.publish(Int32(-1))
        while nhl.raw_data['action_done'].data:
            time.sleep(0.1)

        # the robot is now in a new state
        pos_new, state_new = dhl.get_observation(nhl.raw_data)
        # print('Position: %.2f %.2f' % (pos_new[0], pos_new[1]))
        # print('Direction: %.2f' % dhl.process_direction(nhl.raw_data['pose']))
        # print('Episode: ', episode + 1)
        # print('Step:', step + 1)

        # check whether the agent received the reward
        reward, episode_done, reward_ind = nhl.get_reward(pos, pos_new)
        tot_reward += reward
        # print('Reward:', reward)

        # store in memory
        memory.add_sample((state, action, reward, state_new))

        # train network
        loss = agent.train(memory)
        param.losses_variation[episode].append(loss)
        # print('Loss value:', loss)
        # print('-' * 10)

        # set next state
        state = state_new
        pos = pos_new
        step += 1

        # if episode_done:
        #     print('Reward location found! Reward index:', reward_ind)
        #     print('-' * 10)

    # decay epsilon function
    param.eps = (param.max_eps - param.min_eps) * np.exp(param.decay * episode)

    # store metrics
    dist = np.sum(np.abs(np.array(start_pos) - np.array(param.reward_poses[0])))
    param.rewards_variation.append(tot_reward)
    param.steps_variation.append(step)
    param.target_scores.append(episode_done)
    # param.target_scores.append(dist / float(step) if episode_done else 0.0)

    score = dist / float(step) if episode_done else 0.0
    print('\nepisode: %5d      reward: %6.2f      score: %6.2f      step: %2d      %s\n'
          % (episode + 1, tot_reward, score, step, 'Done' if episode_done else ''))

    return True


def main():
    # initialize agents
    if param.episode:
        str_ = 'double' if param.double else 'single'
        str_ += '_replay' if param.replay else '_noreplay'
        agent = pickle.load(open('results/agent_%s_%d.pkl' % (str_, param.episode), 'rb'))
        memory = pickle.load(open('results/memory_%s_%d.pkl' % (str_, param.episode), 'rb'))
    else:
        agent = Agent(param.n_observations, param.n_actions, param.n_neurons,
                      param.gamma, param.batch_size, param.tau, param.double)
        memory = Memory(50000)

    # if you want to run the experiment from an external program, use VC
    # this will allow you to use frontend interface from python
    vc = VirtualCoach(environment='local', storage_username='nrpuser',
                      storage_password='password')

    # start training
    start = time.time()
    while param.episode < param.episodes:
        # choose initial position
        # start_pos = np.random.rand(2) * 10 + 3
        # start_dir = np.random.rand() * 2 + np.pi
        # print('Starting position: %.2f %.2f' % (start_pos[0], start_pos[1]))
        # print('Starting direction: %.2f' % start_dir)

        start_pos = np.array([8.0, 8.0])

        # with open(folder + 'experiment_configuration.exc', 'r') as file:
        #     data = file.readlines()
        # data[13] = '    <robotPose robotId="husky" x="%.2f" y="%.2f" ' \
        #            'z="0.5" roll="0.0" pitch="-0.0" yaw="%.2f" />\n' \
        #            % (start_pos[0], start_pos[1], start_dir)
        # with open(folder + 'experiment_configuration.exc', 'w') as file:
        #     file.writelines(data)

        # launch experiment
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
        time.sleep(2)

        # inner-loop for running an episode
        done = run_episode(start_pos, param.episode, agent, memory)
        if done:
            param.episode += 1

        # stop experiment
        sim.stop()
        time.sleep(2)

        # save metrics and network for postprocessing
        if (param.episode + 1) % 100 == 0:
            dhl.save_objects(agent, memory, param.episode + 1)

    # print final score
    print('Time elapsed:', time.time() - start)
    print('Target score:', param.target_scores[-1])
    print('Reward visits:', np.sum(param.reward_visits))


if __name__ == '__main__':
    main()
