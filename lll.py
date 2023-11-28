import cv2

# ****************************************
# 拍照 得到校準參數
# ****************************************

def main():
    # 打開網路攝影機
    cap = cv2.VideoCapture(0)  # 如果有多個攝影機，可以嘗試不同的索引，例如1, 2, 等等。
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 352)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 288)

    photo_count = 0 

    while True:
        # 從攝影機中讀取一帧
        ret, frame = cap.read()

        # 確保成功讀取影像
        if not ret:
            break

        # 顯示畫面
        cv2.imshow('Webcam', frame)
        
        
        # 按下 's' 鍵拍照
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # 儲存影像
            cv2.imwrite(f'photo_{photo_count}.jpg', frame)
            print(f"已拍攝 {photo_count} 張照片")
            photo_count += 1

        # 按下 'q' 鍵結束迴圈
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放攝影機資源
    cap.release()

    # 關閉所有視窗
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
