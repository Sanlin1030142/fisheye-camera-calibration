import cv2

def capture_video(index):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error opening video stream: {index}")
    return cap

# 创建两个视频捕获对象
print("cam0")
cap1 = capture_video(0)
print("cam1")
cap2 = capture_video(1)
print("cam2")

# 创建stitcher对象
stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
stitcher.setPanoConfidenceThresh(0.6)  # 设置全景拼接的置信度阈值

# 启用CUDA加速
# stitcher.setUseCUDA(True)


def stitch_frames(*frames):
    # 將兩個影像水平拼接
    stitched_frame = cv2.hconcat(frames)
    return stitched_frame



while cap1.isOpened() and cap2.isOpened() :
    # 从两个视频捕获对象中读取帧
    print( "all open" )
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if ret1 and ret2 :
        # 显示两个相机的视图
        # cv2.imshow('Camera 1', frame1)
        # cv2.imshow('Camera 2', frame2)      

        stitched_frame = stitch_frames(frame1, frame2, frame3)  
        
        # 顯示拼接後的影像
        cv2.imshow('Stitched Frames', stitched_frame)
    else:
        print('Error reading video stream.')

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源并关闭窗口
cap1.release()
cap2.release()
cv2.destroyAllWindows()
