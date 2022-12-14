import rclpy
from threading import Thread
from rclpy.node import Node
import time
from sensor_msgs.msg import Image
from copy import deepcopy
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import Twist, Vector3
import os

def get_file_path():
    """
    Returns a new file path to write to
    """
    # generate next filepath
    dir = "/home/alexiswu/ros2_ws/src/compRobo22-final/tall_camera/callibration"
    i = 0
    while os.path.exists(f"{dir}/callibration{i}.csv"):
        i += 1

    filepath = f"{dir}/callibration{i}.csv"
    return filepath

filepath = get_file_path()

def save_line(filepath,img_coord, floor_coord):
    """
    Save line to CSV file containing one calibration image's data

    Args:
        filepath (string): path to save file
        img_coord (list of double): [x, y]
        floor_coord (list of double): [x, y]
    """
    with open(filepath, 'a') as file:  # Use file to refer to the file object
        file.write(f"{img_coord[0]}, {img_coord[1]}, {floor_coord[0]}, {floor_coord[1]} \n")


class ImageCapture(Node):

    def __init__(self, image_topic):
        """ Initialize the ball tracker """
        super().__init__('video_recording')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        thread = Thread(target=self.loop_wrapper)
        thread.start()
        self.count = 0

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        cv2.namedWindow('video_window')
        while True:
            self.run_loop()
            time.sleep(0.1)










    def find_contour(self,masked_img):
        contours, _ = cv2.findContours(masked_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
        
            # print(type(contours))
            # print(contours)
            return []
        areas = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            areas += [area]
        max_contour = sorted(contours,key=cv2.contourArea, reverse= True)[0]
        return max_contour

    def mask_red_cube(self,frame):
        lower_cube_mask = np.array([130,50,0])
        upper_cube_mask = np.array([255,150,255])
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_cube = cv2.inRange(hsv_image,lower_cube_mask, upper_cube_mask)
        return mask_cube


    def get_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] < 1e-10:
            return 0, 0
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return cX, cY

    def prompt_floor_coords(self):
        """
        Prompt the user for floor coordinates of cube

        Returns:
            list of doubles: Floor coordinates in inches [x.x, y.y]
        """
        prompt = "Input floor coords in inches:\n"
        coord_raw = input(prompt).split()
        return [float(coord_raw[0]), float(coord_raw[1])]

    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        if not self.cv_image is None:
            # frame = cv2.rotate(self.cv_image, cv2.ROTATE_180)
            frame = self.cv_image
            cv2.imshow('video_window', frame)
            filtered_purple = self.mask_red_cube(frame)
            cv2.imshow("filtered_red", filtered_purple)
            k  = cv2.waitKey(1)
    
            max_contour = self.find_contour(filtered_purple)
            if len(max_contour) == 0:
                return
            img_copy = frame.copy()
            cv2.drawContours(img_copy, [max_contour], -1, (0,255,0),2)
            cx, cy = self.get_center(max_contour)
            cv2.circle(img_copy,(cx,cy),7,(0,0,255),-1)

            cv2.imshow("contour", img_copy)

            if k%256  == 32:
                print('space hit, image taken')
                # cv2.imwrite("/home/alexiswu/ros2_ws/src/compRobo22-final/tall_camera/callibration/frame_%d.jpg" %self.count,self.cv_image)
                # self.count += 1
                pixel_coord = [cx,cy]
                world_coord = self.prompt_floor_coords()
                save_line(filepath,pixel_coord,world_coord)
            cv2.waitKey(5)

if __name__ == '__main__':
    node = ImageCapture("/camera/image_raw")
    node.run()


def main(args=None):
    rclpy.init()
    n = ImageCapture("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()


if __name__ == '__main__':
    main()