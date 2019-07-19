import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg


@nrp.MapRobotSubscriber('position', Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapVariable('initial_pose', global_key='initial_pose', initial_value=None)
@nrp.MapVariable('step_index', global_key='step_index', initial_value=0)
@nrp.Neuron2Robot(Topic('/husky/cmd_vel', geometry_msgs.msg.Twist))
def turn_around(t, step_index, position, initial_pose):
    import math
    import rospy
    import numpy as np

    if initial_pose.value is None:
        initial_pose.value = position.value.pose[position.value.name.index('husky')]

    linear = geometry_msgs.msg.Vector3(0, 0, 0)
    angular = geometry_msgs.msg.Vector3(0, 0, 0)
    current_pose = position.value.pose[position.value.name.index('husky')]
    epsilon = .1

    ang = 2.
    lin = 1.

    clientLogger.info(current_pose.position.x, current_pose.position.y)

    def move_up(deg):
        if step_index.value == 0:
            if not (deg < 0 + epsilon or deg > 360 - epsilon):
                angular.z = ang if deg > 180 else -ang
            else:
                initial_pose.value = None
                step_index.value = 1

        elif step_index.value == 1:
            if current_pose.position.x < min(initial_pose.value.position.x + .5, 5):
                linear.x = lin
            else:
                initial_pose.value = None
                step_index.value = 2

    def move_left(deg):
        if step_index.value == 0:
            if not 90 - epsilon < deg < 90 + epsilon:
                angular.z = ang if not 90 < deg < 270 else -ang
            else:
                initial_pose.value = None
                step_index.value = 1

        elif step_index.value == 1:
            if current_pose.position.y < min(initial_pose.value.position.y + .5, 5):
                linear.x = lin
            else:
                initial_pose.value = None
                step_index.value = 2

    def move_down(deg):
        if step_index.value == 0:
            if not 180 - epsilon < deg < 180 + epsilon:
                angular.z = ang if deg < 180 else -ang
            else:
                initial_pose.value = None
                step_index.value = 1

        elif step_index.value == 1:
            if current_pose.position.x > max(initial_pose.value.position.x - .5, 1):
                linear.x = lin
            else:
                initial_pose.value = None
                step_index.value = 2

    def move_right(deg):
        if step_index.value == 0:
            if not 270 - epsilon < deg < 270 + epsilon:
                angular.z = ang if 90 < deg < 270 else -ang
            else:
                initial_pose.value = None
                step_index.value = 1

        elif step_index.value == 1:
            if current_pose.position.y > max(initial_pose.value.position.y - .5, 1):
                linear.x = lin
            else:
                initial_pose.value = None
                step_index.value = 2

    action_done = rospy.get_param('action_done')
    if action_done == 0:
        action = rospy.get_param('action')

        deg = np.arctan2(current_pose.orientation.z, current_pose.orientation.w) * 360 / math.pi
        deg = deg if deg > 0 else 360 + deg

        if action == 0:
            move_up(deg)
        elif action == 1:
            move_left(deg)
        elif action == 2:
            move_down(deg)
        elif action == 3:
            move_right(deg)

        if step_index.value == 2:
            rospy.set_param('robot', [current_pose.position.x, current_pose.position.y])
            rospy.set_param('action_done', 1)
            step_index.value = 0

    return geometry_msgs.msg.Twist(linear=linear, angular=angular)
