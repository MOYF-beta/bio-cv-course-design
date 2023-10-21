import torch
from PIL import Image
import numpy as np

from double_check.state_clasifier import SimpleNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_detect = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5\\runs\\train\\exp2\\weights\\best.pt',
                       # force_reload=True
                       )
weight_doubleCheck = torch.load('net_weights/consep_latest.pth') 
model_doubleCheck = SimpleNet().to(device)
model_doubleCheck.load_state_dict(weight_doubleCheck)

image = Image.open('C:\\Users\\14152\\Desktop\\nuclear-segment\\lizard_yolo\\dataset\\eval\\consep_6.png')
if image.mode == 'RGBA':
    image = image.convert('RGB')

predictions = model_detect(image, size=1080)

boxes = predictions.xyxy[0].cpu().numpy()
scores = boxes[:, 4]
labels = boxes[:, 5]

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1)
image_array = np.array(image)
ax.imshow(image_array)
box_list = []
prop_list = []

for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box[:4]
    width = x2 - x1
    height = y2 - y1

    # double check
    x1_d = x1 - width * 0.05
    x2_d = x2 + width * 0.05
    y1_d = y1 - height * 0.05
    y2_d = y2 + height * 0.05

    subimage = image.crop((x1_d,y1_d,x2_d,y2_d))
    subimage = torch.tensor(np.array(subimage.resize((32, 32))),dtype=torch.float, device=device).transpose(0,2)
    result =model_doubleCheck(subimage).item()
    
    box_list.append(((x1, y1),width,height))
    prop_list.append(result)

max_result = max(prop_list)
min_result = min(prop_list)
range_result = max_result - min_result
prop_list = [(p) / max_result for p in prop_list]

for box,prop in zip(box_list,prop_list):
    color_value = (1 - prop, prop, 0)  # 在RGB颜色空间中插值，红到绿
    rect = patches.Rectangle(box[0],box[1],box[2], linewidth=1, edgecolor=color_value, facecolor='none')
    ax.add_patch(rect)
    
    label_text = f'{score:.2f}'
    
plt.axis('off')
plt.show()