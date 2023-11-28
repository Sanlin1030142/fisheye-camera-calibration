#! /usr/bin/env python

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class Camera():
  
    def __init__( self, DIM, K, D ):
        self.DIM = DIM
        self.K = K
        self.D = D
    
    def remap(self, frame):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K,self.DIM, cv2.CV_16SC2)
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def fisheye_to_equirectangular(self, frame, width_out, height_out):
        equ = np.zeros((height_out,width_out,3), np.uint8)

        width_in = frame.shape[1]
        height_in = frame.shape[0]

        for i in range(height_out):
            for j in range(width_out):
                theta = 2.0 * np.pi * (j / float(width_out - 1)) # artitude
                phi = np.pi * ((2.0 * i / float(height_out - 1)) - 1.0) # vatitude

                x = -np.cos(phi) * np.sin(theta)
                y = np.cos(phi) * np.cos(theta)
                z = np.sin(phi)

                xS = x / (np.abs(x) + np.abs(y))
                yS = y / (np.abs(x) + np.abs(y))

                xi = min(int(0.5 * width_in * (xS + 1.0)), width_in - 1)
                yi = min(int(0.5 * height_in * (yS + 1.0)), height_in - 1)

                equ[i, j, :] = frame[yi, xi, :]

        return equ

    def pipe(self, frame):
      return frame

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('stitched_frame_publisher')

    # Create a publisher to publish stitched_frame data as an Image message
    pub = rospy.Publisher('stitched_frame_topic', Image, queue_size=10)

    # the param Correction the fish eye
    cam_1 = Camera(DIM=(640, 480), K=np.array([[339.94983668649667, 0.0, 313.51323774033034], [0.0, 338.559255378892, 265.57752144550284], [0.0, 0.0, 1.0]]), D=np.array([[-0.022562697866270857], [0.019841725844171796], [-0.026474489579166156], [0.0030526227815419705]]))
    cam_2 = Camera(DIM=(640, 480), K=np.array([[373.8470149373218, 0.0, 320.6197723734125], [0.0, 372.04964046023883, 258.77371651015073], [0.0, 0.0, 1.0]]), D=np.array([[-0.054961170187674324], [-0.0565393452164267], [0.19172051729916142], [-0.17426705462470113]]))
    cam_3 = Camera(DIM=(640, 480), K=np.array([[374.58216418605053, 0.0, 324.99539750258225], [0.0, 372.7834791467761, 273.6591341035029], [0.0, 0.0, 1.0]]), D=np.array([[-0.04432229520634909], [-0.07695785660130959], [0.15721690537848723], [-0.09839313476824274]]))
    cam_4 = Camera(DIM=(640, 480), K=np.array([[377.1016511294628, 0.0, 323.1222883033018], [0.0, 375.52668465664055, 286.8078674299489], [0.0, 0.0, 1.0]]), D=np.array([[-0.04112120133009539], [-0.07124785006697013], [0.13000353909917411], [-0.0908903114922694]]))

    # open the fish eye camara
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
          if not ret0:
              rospy.logerr("Failed to read frame from camera 0")
          
          if not ret1:
              rospy.logerr("Failed to read frame from camera 1")
          
          if not ret2:
              rospy.logerr("Failed to read frame from camera 2")
          
          if not ret3:
              rospy.logerr("Failed to read frame from camera 3")

          break
      
      # Correction
      remap0 = cam_1.remap( frame0 )
      remap1 = cam_2.remap( frame1 )
      remap2 = cam_3.remap( frame2 )
      remap3 = cam_4.remap( frame3 )

      stitched_frame = np.hstack((remap0, remap1, remap2, remap3))

      # show view
      # cv2.imshow('Webcam', stitched_frame)

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
