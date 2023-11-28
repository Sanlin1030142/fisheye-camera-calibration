#! /usr/bin/env python

import cv2
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def main():
    # Initialize ROS node
    rospy.init_node('stitched_frame_publisher')

    # Create a publisher to publish stitched_frame data as an Image message
    pub = rospy.Publisher('stitched_frame_topic', Image, queue_size=10)

    cap0 = cv2.VideoCapture('/dev/video0') 
    cap1 = cv2.VideoCapture('/dev/video2')
    cap2 = cv2.VideoCapture('/dev/video4')
    cap3 = cv2.VideoCapture('/dev/video6')

    # Set the format to MJPG
    cap0.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap1.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap3.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    bridge = CvBridge()

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        # be sure to get the view success
        if not ret0 or not ret1 or not ret2 or not ret3 :    
                break
        
        if not ret0:
            rospy.logerr("Failed to read frame from camera 0")
            

        if not ret1:
            rospy.logerr("Failed to read frame from camera 1")
            

        if not ret2:
            rospy.logerr("Failed to read frame from camera 2")
            

        if not ret3:
            rospy.logerr("Failed to read frame from camera 3")
            

        stitched_frame = np.hstack((frame0, frame3, frame1, frame2))

        # show view
        cv2.imshow('Webcam', stitched_frame)

        # Convert stitched_frame to an Image message
        stitched_frame_msg = bridge.cv2_to_imgmsg( stitched_frame, encoding="bgr8" )

        # Publish stitched_frame as an Image message to the topic
        pub.publish(stitched_frame_msg)

	# press 'q' to end the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap0.release()
    cap1.release()
    cap2.release()
    cap3.release()

    # close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
