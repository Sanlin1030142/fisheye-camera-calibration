import cv2
import numpy as np

# ****************************************
# 簡易的輸出影片
# ****************************************

# 讀取照片
img = cv2.imread('fisheye.jpg')

# 假設我們已經有了校準參數（在實際情況中，需要對您的鏡頭進行校準以獲取這些參數）
# 把在judge.py中得到的參數放進來
DIM=(640, 480)
K=np.array([[366.84408006785185, 0.0, 308.0201095189754], [0.0, 365.39947051963844, 292.456144448516], [0.0, 0.0, 1.0]])
D=np.array([[-0.04308310833515509], [-0.022908647728944992], [0.041677613730846615], [-0.02295071036553948]])

# 計算校正映射
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

# 應用校正映射
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# 顯示校正後的圖像
cv2.imshow('undistorted', undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
