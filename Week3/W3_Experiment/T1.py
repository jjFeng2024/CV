import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 1.读取一张图片，将其转换为HSV空间
image = cv.imread("/Users/feng/Desktop/计算机视觉/Week3/W3_Experiment/test1.jpg")
# 转化为HSV
HSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#显示图片
cv.imshow("Original Image", image)
cv.imshow("HSV Image", HSV)
cv.waitKey(0)
cv.destroyAllWindows()

# 2.分离原图片RGB通道及转换后的图片HSV通道
# 分离RGB
B,G,R = cv.split(image)
# 分离HSV
H,S,V = cv.split(HSV)
# 显示 BGR 通道
cv.imshow("Blue Channel", B)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("Green Channel", G)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("Red Channel", R)
cv.waitKey(0)
cv.destroyAllWindows()

# 显示 HSV 通道
cv.imshow("Hue Channel", H)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("Saturation Channel", S)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imshow("Value Channel", V)
cv.waitKey(0)
cv.destroyAllWindows()

# 3.对RGB三个通道分别画出其三维图（提示：polt_surface函数）
# 创建网格
x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
# image.shape[1]宽，shape[0]长
# 创建一个空白的图形窗口
fig = plt.figure(figsize=(9, 3))

# fig.add_subplot 添加子图 131:一行三列第一个图 projection:指定为三维
# plot_surface 绘制三维曲面图 参数：xyz，cmap:指定颜色映射
# 绘制红色通道的三维曲面
ax = fig.add_subplot(131, projection='3d') # 添加子图
ax.plot_surface(x, y, R, cmap='Reds') # 指定为红色映射
ax.set_title('Red Channel') # 设置标题

# 绘制绿色通道的三维曲面
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(x, y, G, cmap='Greens')
ax.set_title('Green Channel')

# 绘制蓝色通道的三维曲面
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(x, y, B, cmap='Blues')
ax.set_title('Blue Channel')

plt.show() # 显示