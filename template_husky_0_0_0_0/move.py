import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg


@nrp.MapRobotSubscriber('position', Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapRobotSubscriber('camera', Topic('husky/husky/camera', sensor_msgs.msg.Image))
@nrp.MapVariable('initial_pose', global_key='initial_pose', initial_value=None)
@nrp.MapVariable('step_index', global_key='step_index', initial_value=0)
@nrp.MapVariable('direction', global_key='direction', initial_value=0)
@nrp.Neuron2Robot(Topic('/husky/husky/cmd_vel', geometry_msgs.msg.Twist))
def move(t, step_index, position, camera, initial_pose, direction):
    import math
    import rospy
    import numpy as np
    import tf

    if initial_pose.value is None:
        initial_pose.value = position.value.pose[position.value.name.index('husky')]

    linear = geometry_msgs.msg.Vector3(0, 0, 0)
    angular = geometry_msgs.msg.Vector3(0, 0, 0)
    current_pose = position.value.pose[position.value.name.index('husky')]
    directions = [0, math.pi / 2, math.pi, math.pi * 3 / 2]

    # detect red blocks
    image_results = hbp_nrp_cle.tf_framework.tf_lib.detect_red(image=camera.value)
    if image_results.left > .1 or image_results.right > .1:
        rospy.set_param('red', True)
    else:
        rospy.set_param('red', False)

    # set velocity values
    ang = 1.5
    lin = 1.
    epsilon = .017

    def move_forward():
        if step_index.value in [0, 1]:
            # check if robot has moved by 1 meter
            if np.linalg.norm(np.array([current_pose.position.x, current_pose.position.y]) -
                              np.array([initial_pose.value.position.x, initial_pose.value.position.y])) < 1. \
                    and 1.5 <= current_pose.position.x <= 14.5 and 1.5 <= current_pose.position.y <= 14.5:
                linear.x = lin
            else:
                initial_pose.value = None
                step_index.value = 2

    def rotate_and_move(dir):
        d = 1 if dir == 'left' else -1

        if step_index.value == 0:
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

        elif step_index.value == 1:
            move_forward()

    # check if action is done
    action_done = rospy.get_param('action_done')
    if action_done == 0:
        action = rospy.get_param('action')
    else:
        action = -1

    # perform action
    if action == 0:
        move_forward()
    elif action == 1:
        rotate_and_move('left')
    elif action == 2:
        rotate_and_move('right')

    # reset state and send info to main function
    if step_index.value == 2:
        rospy.set_param('direction', direction.value)
        rospy.set_param('position', [current_pose.position.x, current_pose.position.y])

        rospy.set_param('action_done', 1)
        step_index.value = 0

    return geometry_msgs.msg.Twist(linear=linear, angular=angular)
