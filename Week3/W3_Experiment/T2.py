import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# 1.读取home_color图像和灰度图像
home_color = cv.imread("/Users/feng/Desktop/计算机视觉/Week3/W3_Experiment/home_color.jpg")
home_gray = cv.imread("/Users/feng/Desktop/计算机视觉/Week3/W3_Experiment/home_color.jpg", 0)

# 显示彩色图和灰度图（可选）
cv.imshow("home_color", home_color)
cv.imshow("home_gray", home_gray)
cv.waitKey(0)
cv.destroyAllWindows()

# 2&3. 绘制灰度直方图和彩色直方图，并将原图拼接在一起展示
# 第一行左边放灰度直方图,右边放灰度原图;第二行画RGB直方图和原图,以子图形式呈现;全部放在一个窗口里
#创建画布
fig = plt.figure("Gray and Color", figsize=(10, 8))

# 灰度直方图
ax1 = fig.add_subplot(221)
ax1.hist(home_gray.ravel(), 256, [0, 256], color='gray')
ax1.set_title('Gray Histogram')

# 灰度图像
ax2 = fig.add_subplot(222)
ax2.imshow(home_gray, 'gray')
ax2.set_title('Gray Image')
ax2.axis('off')

# 彩色直方图
ax3 = fig.add_subplot(223)
colors = ('b', 'g', 'r')
for i, col in enumerate(colors):
    hist = cv.calcHist([home_color], [i], None, [256], [0, 256])
    ax3.plot(hist, color=col)
ax3.set_title('Color Histogram')

# 彩色图像
ax4 = fig.add_subplot(224)
ax4.imshow(cv.cvtColor(home_color, cv.COLOR_BGR2RGB))
ax4.set_title('Color Image')
ax4.axis('off')

plt.tight_layout()
plt.show()

# 4. 绘制ROI区域的直方图，并拼接原图、mask、ROI区域图像
# 设置 ROI 区域（x：50-100，y：100-200）
x1, x2 = 50, 100
y1, y2 = 100, 200

# 创建掩码
mask = np.zeros(home_gray.shape, dtype=np.uint8)
mask[y1:y2, x1:x2] = 255  # 设为白色区域

# 掩码应用到灰度图上
masked_img = cv.bitwise_and(home_gray, home_gray, mask=mask)

# 直方图计算
hist_full = cv.calcHist([home_gray], [0], None, [256], [0, 256])
hist_mask = cv.calcHist([home_gray], [0], mask, [256], [0, 256])

# 创建窗口
fig2 = plt.figure("ROI and Hist", figsize=(10, 8))
ax1 = fig2.add_subplot(221) #添加子图
ax1.imshow(home_gray, 'gray') #灰度图
ax1.set_title('Gray Image') #设置标题

ax2 = fig2.add_subplot(222)
ax2.imshow(mask,'gray')  # 遮掩图
ax2.set_title('Mark')

ax3 = fig2.add_subplot(223)
ax3.imshow(masked_img, 'gray') #遮掩图
ax3.set_title('Masked Image')

ax4 = fig2.add_subplot(224)
ax4.plot(hist_full), plt.plot(hist_mask) #直方图
ax4.set_title('Hists')

plt.xlim([0,256])
plt.show()
