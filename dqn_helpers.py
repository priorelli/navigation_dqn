import cv2
import numpy as np
import dqn_params as param
import tf
from cv_bridge import CvBridge
import pickle


def process_camera(camera):
    # crop the ROI for removing husky front part
    camera = CvBridge().imgmsg_to_cv2(camera, 'bgr8')[:170, :]
    camera = cv2.resize(camera, (16, 16))
    camera = cv2.cvtColor(camera, cv2.COLOR_BGR2GRAY)
    camera = camera.flatten().astype(np.float32)

    return camera


def process_laser(laser):
    laser = np.asarray(laser.ranges)
    laser[laser == np.inf] = 0.0
    laser = laser[np.arange(0, len(laser), 4)]

    return laser


def process_direction(pose):
    pose = pose.pose[pose.name.index('husky')]

    orientation = [pose.orientation.x, pose.orientation.y,
                   pose.orientation.z, pose.orientation.w]
    _, _, yaw = tf.transformations.euler_from_quaternion(orientation)

    return yaw if yaw >= 0 else 2 * np.pi + yaw


def process_position(pose):
    pose = pose.pose[pose.name.index('husky')]
    position = np.asarray([pose.position.x, pose.position.y])

    return position


def get_observation(raw_data):
    position = process_position(raw_data['pose'])
    direction = process_direction(raw_data['pose'])
    camera = process_camera(raw_data['camera'])
    laser = process_laser(raw_data['laser'])

    input_ = np.concatenate((position / param.n_env, [direction / (2 * np.pi)],
                             camera / 255.0, laser / 5.0))
    # input_ = np.concatenate((position / param.n_env, [direction / (2 * np.pi)]))

    return position, input_[np.newaxis, :]


def save_objects(agent, memory, e):
    str_ = 'double' if param.double else 'single'
    str_ += '_replay' if param.replay else '_noreplay'

    pickle.dump(param.losses_variation, open(param.res_folder + 'losses_variation_%s_%d.pkl' % (str_, e), 'wb'))
    pickle.dump(param.rewards_variation, open(param.res_folder + 'rewards_variation_%s_%d.pkl' % (str_, e), 'wb'))
    pickle.dump(param.steps_variation, open(param.res_folder + 'steps_variation_%s_%d.pkl' % (str_, e), 'wb'))
    pickle.dump(param.target_scores, open(param.res_folder + 'target_scores_%s_%d.pkl' % (str_, e), 'wb'))
    pickle.dump(param.reward_visits, open(param.res_folder + 'reward_visits_%s_%d.pkl' % (str_, e), 'wb'))

    pickle.dump(agent, open(param.res_folder + 'agent_%s_%d.pkl' % (str_, e), 'wb'))
    pickle.dump(memory, open(param.res_folder + 'memory_%s_%d.pkl' % (str_, e), 'wb'))
