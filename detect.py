#!/usr/bin/env python
import rospy
import time
from std_msgs.msg import Bool

from translation import translate
import open3d as o3d
from scipy.spatial import KDTree
from scipy import interpolate
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2


import cv2
import math
import numpy as np
from PIL import Image
import message_filters
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge, CvBridgeError

from yolo import YOLO


class ImageSubscriber:
    def __init__(self, topic_name_1, topic_name_2, msg_type=RosImage):
        self.bridge = CvBridge()
        self.raw_image = np.full((1280, 720, 3), 255)
        # 相机内参
        self.fx = 641.689697265625
        self.fy = 646.1744384765625
        self.cx = 640.9976196289062
        self.cy = 357.24853515625
        self.camera_list = [self.fx, self.fy, self.cx, self.cy]
        self.result_list = []
        self.cv_image = None
        self.cv_depth = None
        self.point_cloud = None
        self.normals = None
        self.save = False

        try:
            self.image_subscriber = message_filters.Subscriber(topic_name_1, msg_type)
            self.depth_subscriber = message_filters.Subscriber(topic_name_2, msg_type)
            self.point_subscriber = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.pointcloud_callback)
            self.image_publisher = rospy.Publisher("/yolo_result/img", msg_type, queue_size=5)
            self.save_subscriber = rospy.Subscriber('/save_list_topic', Bool, self.save_callback)

            ts = message_filters.ApproximateTimeSynchronizer([self.image_subscriber, self.depth_subscriber], 10, 1,
                                                             allow_headerless=True)
            ts.registerCallback(self.call_back)
            rospy.spin()
        except RuntimeError as e:
            raise Exception("cant create subscriber, {}".format(e))

    @staticmethod
    def compute_camera_rotation(normal):
        # 定义相机的目标朝向向量
        target_direction = -normal  # 相机目标朝向与法向量方向相反

        # 定义相机初始朝向向量
        initial_vector = np.array([0, 0, 1])

        # 计算旋转矩阵
        rotation_matrix = Rotation.align_vectors([initial_vector], [target_direction])[0].as_matrix()

        # 将旋转矩阵转换为欧拉角
        euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('zyx', degrees=True)

        return euler_angles[0], euler_angles[1], euler_angles[2]

    @staticmethod
    def convert_to_open3d(pointcloud_msg):
        points = []
        for point in point_cloud2.read_points(pointcloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append(point)
        points = np.array(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    def compute_normals(self):
        # 将 PointCloud2 转换为 Open3D PointCloud
        pcd = self.convert_to_open3d(self.point_cloud)

        # 计算法向量
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # 获取法向量
        normals = np.asarray(pcd.normals)
        self.normals = normals

    def save_list_to_file(self):
        self.compute_normals()
        with open('output.txt', 'w') as file:
            file.write('id' + '\t' + 'real_x' + '\t' + 'real_y' + '\t' + 'depth_value' + '\t' + 'RZRYRX' + '\n')
            for inner_list in self.result_list:
                # inner_list: id, real_x, real_y, depth_value, cx, cy, score
                inner_list[1], inner_list[2], inner_list[3] = translate(inner_list[1], inner_list[2], inner_list[3])
                output_list = inner_list[0:4]
                index = inner_list[5] * self.cv_depth.shape[1] + inner_list[4]  # 计算对应的索引
                if index > len(self.normals):
                    normal = np.array([0, 0, 1])
                else:
                    normal = self.normals[index]  # 获取法向量
                output_list.append(self.compute_camera_rotation(normal))
                for item in output_list:
                    file.write(str(item) + '\t')
                file.write('\n')
        print("List saved to output.txt")

    def save_callback(self, msg):
        if msg.data:
            self.save = msg.data
            self.save_list_to_file()

    def call_back(self, image, depth):
        if not self.save:
            self.cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
            self.cv_depth = self.bridge.imgmsg_to_cv2(depth, "16UC1")
            self.raw_image = self.cv_image.copy()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(self.raw_image, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            img, result_list = yolo.detect_image_pos(frame, self.cv_depth, self.camera_list)
            frame = np.array(img)
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.image_publisher.publish(img_msg)
            self.result_list = result_list

    def pointcloud_callback(self, msg):
        self.point_cloud = msg


if __name__ == "__main__":
    yolo = YOLO()
    crop = False
    count = False

    rospy.init_node('depth_detect', anonymous=True)

    image_subscriber = ImageSubscriber(topic_name_1="camera/color/image_raw",
                                       topic_name_2="/camera/aligned_depth_to_color/image_raw")
