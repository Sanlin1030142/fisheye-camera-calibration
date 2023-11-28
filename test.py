import cv2
import numpy as np

# ****************************************
# 輸出原本跟校正後的影片
# 較準並輸出影片
# ****************************************

# 已校準的參數
# 把在judge.py中得到的參數放進來
DIM=(640, 480)
K=np.array([[361.7752128045716, 0.0, 353.3895106021169], [0.0, 359.86368472710205, 273.607563492823], [0.0, 0.0, 1.0]])
D=np.array([[-0.06182252378110677], [0.046117112255241385], [-0.06156841468456391], [0.030990679298770463]])
# 計算校正映射
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

# 打開攝像頭
cap = cv2.VideoCapture(0)

while(True):
    # 捕捉每一幀圖像
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    cv2.imshow('original_frame', frame)
    # 應用校正映射
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # 顯示校正後的圖像
    cv2.imshow('undistorted', undistorted_frame)

    if cv2.waitKey(1) == ord('q'):
        break

# 釋放攝影鏡頭和銷毀所有開啟的視窗
cap.release()
cv2.destroyAllWindows()