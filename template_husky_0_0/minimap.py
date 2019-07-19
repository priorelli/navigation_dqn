import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg


@nrp.MapRobotSubscriber('position', Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapRobotPublisher('benchmark_metrics', Topic('/benchmark/metrics', sensor_msgs.msg.Image))
def print_map(t, position, benchmark_metrics):
    import cv2

    # display plot
    cv_img = cv2.imread('/home/spock/PycharmProjects/project/plot.png')
    if cv_img is not None:
        msg_frame = CvBridge().cv2_to_imgmsg(cv_img, 'rgb8')
        benchmark_metrics.send_message(msg_frame)