import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg


@nrp.MapRobotSubscriber('position', Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapRobotPublisher('minimap', Topic('/minimap', sensor_msgs.msg.Image))
@nrp.MapRobotPublisher('scores', Topic('/scores', sensor_msgs.msg.Image))
def print_map(t, position, minimap, scores):
    import cv2
    import rospy

    # display plots
    i = rospy.get_param('i')
    cv_img = cv2.imread('/home/spock/PycharmProjects/project/plots/plot_%d.png' % i)
    if cv_img is not None:
        msg_frame = CvBridge().cv2_to_imgmsg(cv_img, 'rgb8')
        minimap.send_message(msg_frame)

    cv_img = cv2.imread('/home/spock/PycharmProjects/project/plots/scores_%d.png' % (i - 1))
    if cv_img is not None:
        msg_frame = CvBridge().cv2_to_imgmsg(cv_img, 'rgb8')
        scores.send_message(msg_frame)
