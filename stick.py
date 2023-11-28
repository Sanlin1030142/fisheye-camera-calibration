import cv2
import numpy as np

# ****************************************
# 最大的程式
# 1. 讀取兩個攝影機
# 2. 重新計算新的相機矩陣(校正後>等距投影)
# 3. 將影格拼接
# 4. 顯示拼接後的影像
# ****************************************


def stitch_frames(*frames):
    # 將兩個影像水平拼接
    stitched_frame = np.hstack(frames)
    return stitched_frame

def capture_video(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error opening video stream: {index}")
    return cap

class Camera:
    def __init__(self, DIM, K, D):
        self.DIM=DIM
        self.K=K
        self.D=D

    def remap(self, frame):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.K,self.DIM, cv2.CV_16SC2)
        return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def fisheye_to_equirectangular(self, frame, width_out, height_out):
        # 建立輸出圖像
        equ = np.zeros((height_out,width_out,3), np.uint8)

        width_in = frame.shape[1]
        height_in = frame.shape[0]

        # 對每個像素進行投影
        for i in range(height_out):
            for j in range(width_out):
                theta = 2.0 * np.pi * (j / float(width_out - 1)) # 經度
                phi = np.pi * ((2.0 * i / float(height_out - 1)) - 1.0) # 緯度

                x = -np.cos(phi) * np.sin(theta)
                y = np.cos(phi) * np.cos(theta)
                z = np.sin(phi)

                xS = x / (np.abs(x) + np.abs(y))
                yS = y / (np.abs(x) + np.abs(y))

                xi = min(int(0.5 * width_in * (xS + 1.0)), width_in - 1)
                yi = min(int(0.5 * height_in * (yS + 1.0)), height_in - 1)

                equ[i, j, :] = frame[yi, xi, :]

        return equ
    
    def pipe( self, frame ) :
        return frame
        reframe = self.remap( frame )
        reframe = self.fisheye_to_equirectangular( reframe, 640, 480 )
        return reframe


if __name__ == "__main__":
    # 給兩個class 參數
    cam_1 = Camera(DIM=(640, 480), K=np.array([[339.94983668649667, 0.0, 313.51323774033034], [0.0, 338.559255378892, 265.57752144550284], [0.0, 0.0, 1.0]]), D=np.array([[-0.022562697866270857], [0.019841725844171796], [-0.026474489579166156], [0.0030526227815419705]]))
    cam_2 = Camera(DIM=(640, 480), K=np.array([[373.8470149373218, 0.0, 320.6197723734125], [0.0, 372.04964046023883, 258.77371651015073], [0.0, 0.0, 1.0]]), D=np.array([[-0.054961170187674324], [-0.0565393452164267], [0.19172051729916142], [-0.17426705462470113]]))
    cam_3 = Camera(DIM=(640, 480), K=np.array([[374.58216418605053, 0.0, 324.99539750258225], [0.0, 372.7834791467761, 273.6591341035029], [0.0, 0.0, 1.0]]), D=np.array([[-0.04432229520634909], [-0.07695785660130959], [0.15721690537848723], [-0.09839313476824274]]))
    # cam_4 = Camera(DIM=(640, 480), K=np.array([[377.1016511294628, 0.0, 323.1222883033018], [0.0, 375.52668465664055, 286.8078674299489], [0.0, 0.0, 1.0]]), D=np.array([[-0.04112120133009539], [-0.07124785006697013], [0.13000353909917411], [-0.0908903114922694]]))
    # 開啟兩個攝影機
    print("cam0")
    cap1 = capture_video(0)
    print("cam1")
    cap2 = capture_video(1)
    print("cam2")
    cap3 = capture_video(2)

    # 创建stitcher对象
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    stitcher.setPanoConfidenceThresh(0.6)  # 设置全景拼接的置信度阈值


    while cap1.isOpened() and cap2.isOpened() and cap3.isOpened():

        # 成功讀取，ret為True，frame為獲取到的影像
        print( "all open" )
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        # ret4, frame4 = cap4.read()

        if ret1 and ret2 :
            # 重新計算新的相機矩陣
            reframe1 = cam_1.pipe(frame1)
            reframe2 = cam_2.pipe(frame2)
            reframe3 = cam_3.pipe(frame3)
            # reframe4 = cam_4.pipe(frame4)

            
            # 將影格拼接
            stitched_frame = stitch_frames( reframe1, reframe2, reframe3 )
            
            # 顯示拼接後的影像
            cv2.imshow('reframe1', stitched_frame )

        else:
            print('Error reading video stream.')

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源并关闭窗口
    cap1.release()
    cap2.release()
    cap3.release()
    cv2.destroyAllWindows()
