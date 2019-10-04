import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import std_msgs.msg
import geometry_msgs.msg


@nrp.MapRobotSubscriber('action_topic', Topic('action_topic', std_msgs.msg.Int32))
@nrp.MapRobotPublisher('action_done_topic', Topic('action_done_topic', std_msgs.msg.Bool))
@nrp.MapRobotPublisher('direction_topic', Topic('direction_topic', std_msgs.msg.Int32))
@nrp.MapRobotSubscriber('position', Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapVariable('initial_pose', global_key='initial_pose', initial_value=None)
@nrp.MapVariable('step_index', global_key='step_index', initial_value=0)
@nrp.MapVariable('direction', global_key='direction', initial_value=0)
@nrp.Neuron2Robot(Topic('/husky/husky/cmd_vel', geometry_msgs.msg.Twist))
def move(t, step_index, position, initial_pose, direction_topic, direction,
         action_topic, action_done_topic):
    import math
    import numpy as np
    import tf

    linear = geometry_msgs.msg.Vector3(0, 0, 0)
    angular = geometry_msgs.msg.Vector3(0, 0, 0)

    direction_topic.send_message(std_msgs.msg.Int32(direction.value))

    if action_topic.value is None or action_topic.value.data < 0:
        action_done_topic.send_message(std_msgs.msg.Bool(False))
        return geometry_msgs.msg.Twist(linear=linear, angular=angular)

    if initial_pose.value is None:
        initial_pose.value = position.value.pose[position.value.name.index('husky')]

    current_pose = position.value.pose[position.value.name.index('husky')]
    directions = [0, math.pi / 2, math.pi, math.pi * 3 / 2]

    # set velocity values
    ang = 1.0
    lin = 1.0
    epsilon = .012

    def move_forward():
        # check if robot has moved by 1 meter
        if np.linalg.norm(np.array([current_pose.position.x, current_pose.position.y]) -
                          np.array([initial_pose.value.position.x, initial_pose.value.position.y])) < 1.0 \
                and 1.5 <= current_pose.position.x <= 13.5 and 1.5 <= current_pose.position.y <= 13.5:
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

        # compute final direction
        final_dir = (direction.value + d) % 4

        # check if robot has rotated by 90 degrees
        if not directions[final_dir] - epsilon < yaw < directions[final_dir] + epsilon:
            angular.z = ang * d
        else:
            direction.value = final_dir
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

