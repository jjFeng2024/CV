# 1、读取一段本地视频（Waymo.mp4）并播放
import cv2

# 读取本地视频mp4.文件
video_path="/Users/feng/Desktop/计算机视觉/Week1/W1_Experiment/Waymo.mp4"
cap = cv2.VideoCapture(video_path)

# 异常处理
if not cap.isOpened():
    print("无法打开视频，请检查路径！")
    exit()

# 循环读取并播放视频
while True:
    ret, frame = cap.read()  # 读取视频的每一帧
    if not ret:
        print("视频播放结束或读取失败")
        break

    cv2.imshow("Video Player", frame)  # 显示当前帧

    # 按 'q' 退出播放
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()