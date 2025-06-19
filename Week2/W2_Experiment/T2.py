import cv2
import numpy as np

# 读取灰度图片
my_photo = cv2.imread('/Users/feng/Desktop/计算机视觉/Week2/W2_Experiment/Leopard.png', 0)
# 获取高和宽得到图片的大小
height, width = my_photo.shape[:2]

# 显示图片
cv2.imshow('Gray Image', my_photo)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ------------------------------------------------缩放正方形--------------------------------------------------
# 计算比例
scale_up_x = 600/width
scale_up_y = 600/height

image = cv2.resize(my_photo, None, fx=scale_up_x, fy=scale_up_y, interpolation=cv2.INTER_LINEAR)

# 显示图像
cv2.imshow('Scale Down to 600*600.', image)
cv2.waitKey()
cv2.destroyAllWindows()

# ------------------------------------------------切片--------------------------------------------------
# 修改长宽
width = 600
height = 600
# 创建一个与图像大小相同的黑色背景
mask = np.zeros((height, width), dtype=np.uint8)

# 定义圆心坐标和半径（这里假设圆心在图像的中心）
center = (width // 2, height // 2)
radius = min(width, height) // 2

# 在掩膜上绘制白色圆形
cv2.circle(mask, center, radius, 255, -1)

# 对图像应用掩膜
masked_image = cv2.bitwise_and(image, image, mask=mask)

# 显示原始图像和掩膜图像
cv2.imshow('Original Image', image)
cv2.imshow('Masked Image', masked_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存图片
cv2.imwrite('Masked_Image.jpg', masked_image)