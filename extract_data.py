import roslib
import rosbag
import rospy
import cv2
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
 
rgb = '/home/liwei/catkin_ws/src/crack_ros/dataset/rgb/'  #rgb path
depth = '/home/liwei/catkin_ws/src/crack_ros/dataset/depth/'   #depth path
bridge = CvBridge()
 
file_handle1 = open('/home/liwei/catkin_ws/src/crack_ros/dataset/depth.txt', 'w')
file_handle2 = open('/home/liwei/catkin_ws/src/crack_ros/dataset/rgb.txt', 'w')
 
with rosbag.Bag('/home/liwei/catkin_ws/src/crack_ros/dataset/defect/5.bag', 'r') as bag:
    i = 0
    for topic,msg,t in bag.read_messages():
        if topic == "/camera/aligned_depth_to_color/image_raw":  #depth topic
            cv_image = bridge.imgmsg_to_cv2(msg,"16UC1")
            timestr = "%.6f" %  msg.header.stamp.to_sec()   #depth time stamp
            image_name = timestr+ ".png"
            path = "depth/" + image_name
            file_handle1.write(timestr + " " + path + '\n')
            cv2.imwrite(depth + image_name, cv_image)
        if topic == "/camera/color/image_raw":   #rgb topic
            cv_image = bridge.imgmsg_to_cv2(msg,"bgr8")
            timestr = "%.6f" %  msg.header.stamp.to_sec()   #rgb time stamp
            image_name = timestr+ ".png"
            path = "rgb/" + image_name
            if i==10:
            	file_handle2.write(timestr + " " + path + '\n')
            	cv2.imwrite(rgb + image_name, cv_image)
            	i = 0
            i = i + 1
            
file_handle1.close()
file_handle2.close()
