import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import std_msgs.msg
import geometry_msgs.msg
import sensor_msgs.msg
import gazebo_msgs.msg


@nrp.MapRobotSubscriber('action_topic', Topic('action_topic', std_msgs.msg.Int32))
@nrp.MapRobotPublisher('action_done_topic', Topic('action_done_topic', std_msgs.msg.Bool))
@nrp.MapRobotSubscriber('position', Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapRobotSubscriber('laser', Topic('/husky/husky/laser/scan', sensor_msgs.msg.LaserScan))
@nrp.MapVariable('initial_pose', global_key='initial_pose', initial_value=None)
@nrp.MapVariable('step_index', global_key='step_index', initial_value=0)
@nrp.Neuron2Robot(Topic('/husky/husky/cmd_vel', geometry_msgs.msg.Twist))
def move(t, step_index, position, laser, initial_pose, action_topic, action_done_topic):
    import math
    import numpy as np
    import tf

    if position.value is None:
        return

    linear = geometry_msgs.msg.Vector3(0, 0, 0)
    angular = geometry_msgs.msg.Vector3(0, 0, 0)

    if initial_pose.value is None:
        initial_pose.value = position.value.pose[position.value.name.index('husky')]

    current_pose = position.value.pose[position.value.name.index('husky')]

    if action_topic.value is None or action_topic.value.data < 0:
        action_done_topic.send_message(std_msgs.msg.Bool(False))
        return geometry_msgs.msg.Twist(linear=linear, angular=angular)

    # set velocity values
    ang = 1.2
    lin = 1.0
    epsilon = .015

    def move_forward():
        # check if robot has moved by 1 meter
        if np.linalg.norm(np.array([current_pose.position.x, current_pose.position.y]) -
                          np.array([initial_pose.value.position.x, initial_pose.value.position.y])) < 1.0 \
                and min(np.asarray(laser.value.ranges)[140:220]) > 1.0:
            linear.x = lin
        else:
            initial_pose.value = None
            step_index.value = 0
            action_done_topic.send_message(std_msgs.msg.Bool(True))

    def rotate(dir_):
        d = 1 if dir_ == 'left' else -1

        # get orientation of robot
        orientation = [current_pose.orientation.x, current_pose.orientation.y,
                       current_pose.orientation.z, current_pose.orientation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(orientation)
        yaw = yaw if yaw >= 0 else 2 * math.pi + yaw

        orientation_old = [initial_pose.value.orientation.x, initial_pose.value.orientation.y,
                           initial_pose.value.orientation.z, initial_pose.value.orientation.w]
        _, _, yaw_old = tf.transformations.euler_from_quaternion(orientation_old)
        yaw_new = yaw_old + d * (math.pi / 2)
        yaw_new = yaw_new if yaw_new >= 0 else 2 * math.pi + yaw_new

        # check if robot has rotated by 90 degrees
        if not yaw_new - epsilon < yaw < yaw_new + epsilon:
            angular.z = ang * d
        else:
            initial_pose.value = None
            step_index.value = 1

    # perform action
    if action_topic.value.data == 0:
        move_forward()
    elif action_topic.value.data == 1:
        if step_index.value == 0:
            rotate('left')
        else:
            move_forward()
    elif action_topic.value.data == 2:
        if step_index.value == 0:
            rotate('right')
        else:
            move_forward()

    return geometry_msgs.msg.Twist(linear=linear, angular=angular)
