import hbp_nrp_cle.tf_framework as nrp
from hbp_nrp_cle.robotsim.RobotInterface import Topic
import geometry_msgs.msg


@nrp.MapRobotSubscriber('position', Topic('/gazebo/model_states', gazebo_msgs.msg.ModelStates))
@nrp.MapRobotPublisher('benchmark_metrics', Topic('/benchmark/metrics', sensor_msgs.msg.Image))
def print_map(t, position, benchmark_metrics):
    import numpy as np
    # import matplotlib.pyplot as plt
    import cv2

    cv_img = cv2.imread('/home/spock/PycharmProjects/project/robot.png')
    if cv_img is not None:
        msg_frame = CvBridge().cv2_to_imgmsg(cv_img, 'rgb8')
        benchmark_metrics.send_message(msg_frame)

    # fig = plt.figure()
    # plot = fig.add_subplot(111)
    # x = np.arange(0, 100, 0.1)
    # y = np.sin(x) / x
    # plot.plot(x, y)

    # fig.canvas.draw()

    # w, h = fig.canvas.get_width_height()
    # cv_img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8').reshape(w, h, 3)
    # plt.close()

