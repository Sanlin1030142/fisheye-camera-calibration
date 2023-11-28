#!/usr/bin/env python

import cv2
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray

def main():
    # Initialize ROS node
    rospy.init_node('stitched_frame_publisher')

    # Create a publisher to publish stitched_frame data
    pub = rospy.Publisher('stitched_frame_topic', Float32MultiArray, queue_size=10)

    cap0 = cv2.VideoCapture('/dev/video0') 
    cap1 = cv2.VideoCapture('/dev/video2')
    cap2 = cv2.VideoCapture('/dev/video4')
    cap3 = cv2.VideoCapture('/dev/video7')

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        # be sure to get the view success
        if not ret0 and not ret1 and not ret2 and not ret3:
            print("one camera is dead" )
            break

        stitched_frame = np.hstack((frame0, frame1, frame2, frame3))

        # show view
        cv2.imshow('Webcam', stitched_frame)

        # Convert stitched_frame to a format that can be published as a message
        stitched_frame_msg = Float32MultiArray(data=stitched_frame.flatten())

        # Publish stitched_frame to the topic
        pub.publish(stitched_frame_msg)

        # press 'q' to end the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # release camera 
    cap0.release()
    cap1.release()
    cap2.release()
    cap3.release()

    # close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
