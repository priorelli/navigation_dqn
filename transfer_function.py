import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg


@nrp.MapRobotSubscriber('position', Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapVariable('initial_pose', global_key='initial_pose', initial_value=None)
@nrp.MapVariable('step_index', global_key='step_index', initial_value=0)
@nrp.Neuron2Robot(Topic('/husky/husky/cmd_vel', geometry_msgs.msg.Twist))
def move(t, step_index, position, initial_pose):
    import math
    import rospy
    import numpy as np
    import tf

    if initial_pose.value is None:
        initial_pose.value = position.value.pose[position.value.name.index('husky')]

    linear = geometry_msgs.msg.Vector3(0, 0, 0)
    angular = geometry_msgs.msg.Vector3(0, 0, 0)
    current_pose = position.value.pose[position.value.name.index('husky')]

    ang = 1.
    lin = 1.
    epsilon = .015

    def move_forward():
        if step_index.value == 0:
            if np.linalg.norm(np.array([current_pose.position.x, current_pose.position.y]) -
                              np.array([initial_pose.value.position.x, initial_pose.value.position.y])) < .5:
                linear.x = lin
            else:
                initial_pose.value = None
                step_index.value = 1

    def turn_left():
        if step_index.value == 0:
            initial_orientation = [initial_pose.value.orientation.x, initial_pose.value.orientation.y,
                                   initial_pose.value.orientation.z, initial_pose.value.orientation.w]
            _, _, initial_yaw = tf.transformations.euler_from_quaternion(initial_orientation)
            initial_yaw = initial_yaw if initial_yaw >= 0 else 2 * math.pi + initial_yaw
            final_yaw = (initial_yaw + math.pi / 2) % (2 * math.pi)

            orientation = [current_pose.orientation.x, current_pose.orientation.y,
                           current_pose.orientation.z, current_pose.orientation.w]
            _, _, yaw = tf.transformations.euler_from_quaternion(orientation)
            yaw = yaw if yaw >= 0 else 2 * math.pi + yaw

            clientLogger.info(yaw, final_yaw)

            if not final_yaw - epsilon < yaw < final_yaw + epsilon:
                angular.z = ang
            else:
                initial_pose.value = None
                step_index.value = 1

    def turn_right():
        if step_index.value == 0:
            initial_orientation = [initial_pose.value.orientation.x, initial_pose.value.orientation.y,
                                   initial_pose.value.orientation.z, initial_pose.value.orientation.w]
            _, _, initial_yaw = tf.transformations.euler_from_quaternion(initial_orientation)
            initial_yaw = initial_yaw if initial_yaw >= 0 else 2 * math.pi + initial_yaw
            final_yaw = (initial_yaw - math.pi / 2) % (2 * math.pi)

            orientation = [current_pose.orientation.x, current_pose.orientation.y,
                           current_pose.orientation.z, current_pose.orientation.w]
            _, _, yaw = tf.transformations.euler_from_quaternion(orientation)
            yaw = yaw if yaw >= 0 else 2 * math.pi + yaw

            clientLogger.info(yaw, final_yaw)

            if not final_yaw - epsilon < yaw < final_yaw + epsilon:
                angular.z = -ang
            else:
                initial_pose.value = None
                step_index.value = 1

    action_done = rospy.get_param('action_done')
    if action_done == 0:
        action = rospy.get_param('action')
    else:
        action = 0

    if action == 1:
        move_forward()
    elif action == 2:
        turn_left()
    elif action == 3:
        turn_right()

    if step_index.value == 1:
        # orientation = [current_pose.orientation.x, current_pose.orientation.y,
        #                current_pose.orientation.z, current_pose.orientation.w]
        # _, _, yaw = tf.transformations.euler_from_quaternion(orientation)
        # yaw = yaw if yaw >= 0 else 2 * math.pi + yaw
        # rospy.set_param('orientation', yaw)
        # rospy.set_param('position', [current_pose.position.x, current_pose.position.y])

        rospy.set_param('action_done', 1)
        step_index.value = 0

    return geometry_msgs.msg.Twist(linear=linear, angular=angular)
