from PIL import Image, ImageDraw
import os

path = "C:\\Users\\14152\\Desktop\\nuclear-segment\\dataset\\Lizard_Images"
name = "consep_6"

# 打开图像文件
image = Image.open(os.path.join(path, f"{name}.png"))

# 打开标签文件
with open(os.path.join(path, f"{name}.txt"), 'r') as label_file:
    labels = label_file.readlines()

# 创建一个绘图对象
draw = ImageDraw.Draw(image)

for label in labels:
    # 解析标签信息，假设标签格式是：class x_center y_center width height
    label_info = label.strip().split(' ')
    class_name = label_info[0]
    x_center = float(label_info[1])
    y_center = float(label_info[2])
    width = float(label_info[3])
    height = float(label_info[4])

    # 计算目标的左上角和右下角坐标
    x_min = int((x_center - width/2) * image.width)
    y_min = int((y_center - height/2) * image.height)
    x_max = int((x_center + width/2) * image.width)
    y_max = int((y_center + height/2) * image.height)

    # 画出边框
    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red")

# 显示图像
image.show()
