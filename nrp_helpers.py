import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from gazebo_msgs.msg import ModelStates
import dqn_params as param


raw_data = {'camera': None, 'laser': None, 'pose': None, 'action_done': None}


# for the callbacks below you do not need to put in to the nrp tf.
# get data from rostopic
def get_data(data, args):
    args[0][args[1]] = data


# subscribe to ros topics
def perform_subscribers():
    rospy.Subscriber('/husky/husky/camera', Image, get_data,
                     callback_args=[raw_data, 'camera'])
    rospy.Subscriber('/husky/husky/laser/scan', LaserScan, get_data,
                     callback_args=[raw_data, 'laser'])
    rospy.Subscriber('/gazebo/model_states', ModelStates, get_data,
                     callback_args=[raw_data, 'pose'])
    rospy.Subscriber('action_done_topic', Bool, get_data,
                     callback_args=[raw_data, 'action_done'])


def get_action_name(action_id):
    return 'Move forward' if action_id == 0 else 'Turn left' \
           if action_id == 1 else 'Turn right' if action_id == 2 else ''


def get_reward(pos, next_pos):
    if np.linalg.norm(pos - next_pos) < 0.1:  # safety factor
        return param.reward_obstacle, 0, None
    else:
        for r_idx, r_pos in enumerate(param.reward_poses):
            if np.linalg.norm(next_pos - r_pos) <= 1.0:
                param.reward_visits[int(r_pos[0]), int(r_pos[1])] += 1
                return param.reward_target_found, 1, r_idx
    return param.reward_free, 0, None
