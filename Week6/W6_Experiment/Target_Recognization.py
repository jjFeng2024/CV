import cv2
import os
import numpy as np

# 加载人脸分类器
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# 加载AdaBoost分类器
boost = cv2.ml.Boost_create()

# 提取Haar特征
def extract_haar_features(image, is_roi=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if not is_roi:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            gray = gray[y:y+h, x:x+w]
        # 如果没有检测到人脸，就保留整图 gray，用于负样本特征提取
    roi = cv2.resize(gray, (24, 24))
    hog = cv2.HOGDescriptor((24, 24), (8, 8), (4, 4), (8, 8), 9)
    feature = hog.compute(roi)
    return True, feature.flatten()
# 提取正样本和负样本的特征
pos_features = []
neg_features = []

# 准备正样本特征
pos_samples_folder = '/Users/feng/Desktop/计算机视觉/Week5/W5_Experiment/yale_face'
for filename in os.listdir(pos_samples_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'JPG', 'bmp')):
        image_path = os.path.join(pos_samples_folder, filename)
        image = cv2.imread(image_path)
        has_face, feature = extract_haar_features(image, is_roi=False)
        if has_face:
            pos_features.append(feature)

# 准备负样本特征
neg_samples_folder = '/Users/feng/Desktop/计算机视觉/Week5/W5_Experiment/non_face'
for filename in os.listdir(neg_samples_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'JPG')):
        image_path = os.path.join(neg_samples_folder, filename)
        image = cv2.imread(image_path)
        has_face, feature = extract_haar_features(image, is_roi=False)
        neg_features.append(feature)

# 准备训练数据
features = np.vstack((np.vstack(pos_features), np.vstack(neg_features)))
labels = np.hstack((np.ones(len(pos_features)), np.zeros(len(neg_features)))).astype(np.int32)

# 训练AdaBoost模型
boost.train(features, cv2.ml.ROW_SAMPLE, labels)

folder_path = '/Users/feng/Desktop/计算机视觉/Week5/W5_Experiment/test_photo'
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png', 'JPG')):
        # 读取图像
        image_path = os.path.join(folder_path, filename)
        frame = cv2.imread(image_path)

        size = frame.shape[:2]
        minSize_1 = (size[1] // 10, size[0] // 10)  # 计算最小尺寸
        face_rects = face_cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=2,
                                                   minSize=minSize_1)
        # 在图像上绘制检测到的人脸
        for (x, y, w, h) in face_rects:
            # 提取当前人脸区域的特征
            roi = frame[y:y+h, x:x+w]
            has_face, feature = extract_haar_features(roi, is_roi=True)
            # 使用AdaBoost模型进行预测
            _, result = boost.predict(feature.reshape(1, -1))
            if result == 1:  # 1 表示检测到人脸
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        cv2.imshow('detection', frame)
        cv2.waitKey(0)

cv2.destroyAllWindows()