import cv2 as cv
from matplotlib import pyplot as plt

# 1. 读取原图并转换为灰度图
image = cv.imread("/Users/feng/Desktop/计算机视觉/Week3/W3_Experiment/test1.jpg")
gray = cv.imread("/Users/feng/Desktop/计算机视觉/Week3/W3_Experiment/test1.jpg", 0)

# 2. 直方图均衡化
equalized = cv.equalizeHist(gray)

# 3. 显示图像对比
cv.imshow("Original Gray", gray)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imshow("Equalized Gray", equalized)
cv.waitKey(0)
cv.destroyAllWindows()

# 4. 显示直方图对比（用 matplotlib）
plt.figure(figsize=(8, 4))
plt.plot(cv.calcHist([gray], [0], None, [256], [0, 256]), color='blue', label='Original')
plt.plot(cv.calcHist([equalized], [0], None, [256], [0, 256]), color='red', label='Equalized')

plt.title('Gray Histogram Comparison')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.legend()
plt.xlim([0, 256])
plt.grid(True)
plt.tight_layout()
plt.show()
