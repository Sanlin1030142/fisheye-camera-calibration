import cv2

# ****************************************
# 同時輸出所有(4)的攝影機
# ****************************************

def main():
    # 打開網路攝影機
    cap0 = cv2.VideoCapture(0)  # 如果有多個攝影機，可以嘗試不同的索引，例如1, 2, 等等。
    cap1 = cv2.VideoCapture(1)
    cap2 = cv2.VideoCapture(2)
    cap3 = cv2.VideoCapture(3)

    while True:
        # 從攝影機中讀取一帧
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()

        # 確保成功讀取影像
        if not ret0 and not ret1 and not ret2 and not ret3:
            break

        # 顯示畫面
        cv2.imshow('Webcam', frame0)
        cv2.imshow('Webcam1', frame1)
        cv2.imshow('Webcam2', frame2)
        cv2.imshow('Webcam3', frame3)
        


        # 按下 'q' 鍵結束迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放攝影機資源
    cap0.release()
    cap1.release()
    cap2.release()
    cap3.release()

    # 關閉所有視窗
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
