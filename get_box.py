import torch
import numpy as np
from PIL import Image
from skimage import io, color, filters, img_as_ubyte

from double_check.state_clasifier import SimpleNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_detect = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='yolov5\\runs\\train\\exp2\\weights\\best.pt',
                       # force_reload=True
                       )
weight_doubleCheck = torch.load('./net_weights/._latest.pth') 
model_doubleCheck = SimpleNet().to(device)
model_doubleCheck.load_state_dict(weight_doubleCheck)

# 必须在加载yolov5后加载plt，因为ultralytics不知道在它的库里写了啥会使得plt无法显示图窗
import matplotlib; matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def segment(image):
    image_gray = color.rgb2gray(np.array(image))
    threshold_value = filters.threshold_otsu(image_gray)
    binary_mask = image_gray > threshold_value
    binary_mask = img_as_ubyte(binary_mask)
    output_mask = np.zeros_like(image_gray)
    output_mask[binary_mask > 0] = 1
    return output_mask




def predict(
        im_path = 'C:\\Users\\14152\\Desktop\\nuclear-segment\\lizard_yolo\\dataset\\eval\\consep_6.png',
        segment_method = segment,
        cell_mask = 1
        ):

    image = Image.open(im_path)
    masks = np.zeros((image.height,image.width))
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    predictions = model_detect(image, size=1080)

    boxes = predictions.xyxy[0].cpu().numpy()
    scores = boxes[:, 4]
    labels = boxes[:, 5]



    box_list = []
    prop_list = []

    #### 子图像图像分割 ####
    index = 0
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box[:4]
        width = x2 - x1
        height = y2 - y1

        x1_d = int(x1 - width * 0.05)
        x2_d = int(x2 + width * 0.05)
        y1_d = int(y1 - height * 0.05)
        y2_d = int(y2 + height * 0.05)

        subimage = image.crop((x1_d,y1_d,x2_d,y2_d))
        subimage_mask = segment_method(subimage)
        subimage_mask[subimage_mask==cell_mask] = index
        index = index + 1
        masks[y1_d:y2_d, x1_d:x2_d] = subimage_mask
        subimage_torch = torch.tensor(np.array(subimage.resize((32, 32))),dtype=torch.float, device=device).transpose(0,2)
        result =model_doubleCheck(subimage_torch).item()
        

        box_list.append(((x1, y1),width,height))
        prop_list.append(result)

    #### 结果可视化 ####
    max_result = max(prop_list)
    min_result = min(prop_list)
    range_result = max_result - min_result
    prop_list = [(p) / max_result for p in prop_list]

    fig, (ax1,ax2) = plt.subplots(1,2)
    image_array = np.array(image)
    ax1.imshow(image_array)
    for box,prop in zip(box_list,prop_list):
        color_value = (1 - prop, prop, 0)
        rect = patches.Rectangle(box[0],box[1],box[2], linewidth=1, edgecolor=color_value, facecolor='none')
        ax1.add_patch(rect)
    ax1.axis('off')
    ax1.set_title('Boxes') 

    ax2.imshow(masks)
    ax2.axis('off')
    ax2.set_title('Masks')

    plt.tight_layout()
    plt.show()

    return masks

predict(cell_mask = 0)