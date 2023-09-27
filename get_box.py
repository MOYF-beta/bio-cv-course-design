import torch
from PIL import Image
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5\\runs\\drone\\best.pt'
                       )

image = Image.open('C:\\Users\\14152\\Desktop\\drone\\dataset_2\\images\\138.jpg')
if image.mode == 'RGBA':
    image = image.convert('RGB')

predictions = model(image, size=640)

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

for box, score, label in zip(boxes, scores, labels):
    x1, y1, x2, y2 = box[:4]
    width = x2 - x1
    height = y2 - y1
    
    rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    
    label_text = f'{score:.2f}'
    ax.text(x1, y1, label_text, bbox=dict(facecolor='white', alpha=0.5))
    
plt.axis('off')
plt.show()