import rospy
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge

import cv2
import math
import numpy as np
from PIL import Image

from yolo import YOLO

from sensor_msgs.msg import PointCloud2


class CompressedImageSubscriber:
    def __init__(self, topic_name):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic_name, CompressedImage, self.callback)
        self.image_publisher = rospy.Publisher("/yolo_result/img", ROSImage, queue_size=5)
        self.yolo = YOLO()
        self.save_path = "./kitti_result/"

    def callback(self, data):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
        current_time = data.header.stamp
        print(current_time)
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))
        # 进行检测
        img = self.yolo.detect_image_in_kitti(frame, self.save_path, current_time)
        frame = np.array(img)
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        self.image_publisher.publish(img_msg)


if __name__ == '__main__':
    rospy.init_node('compressed_image_subscriber', anonymous=True)
    topic_name = '/camera/image_color/compressed'
    compressed_image_subscriber = CompressedImageSubscriber(topic_name)
    rospy.spin()
