# 1、读取一张图片并显示；2、在图片中加入文字（学号+姓名）；3、保存该图片到本地。
import cv2
import numpy as np
from PIL import ImageDraw, Image, ImageFont

# 读取并显示图片
image_path = "/Users/feng/Desktop/海.jpg"  #图片路径设置
image = cv2.imread(image_path)
cv2.imshow("Original Image", image)
cv2.waitKey(0)  # 等待用户按键
cv2.destroyAllWindows()

# 在图片上添加文字
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(pil_image)
font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"
font = ImageFont.truetype(font_path, 200)
text = "23122721 冯俊佳"  # 学号和姓名
position = (1300, 300)  # 文字位置
color = (0, 0, 0)  # 文字颜色
draw.text(position, text, fill=color, font=font)

# 保存图片
output_path = "海_edited.jpg"
pil_image.save(output_path)
print(f"图片已保存至 {output_path}")

# 重新显示修改后的图片
cv2.imshow("Modified Image", cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()