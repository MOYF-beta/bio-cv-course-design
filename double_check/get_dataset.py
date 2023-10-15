from matplotlib import patches
import scipy.io as sio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
num_true = 100
def visualize_bboxes(image_path, bbox_list):
    # Load image
    image = plt.imread(image_path)
    plt.imshow(image)
    
    # Plot bounding boxes
    for bbox in bbox_list:
        y1, y2, x1, x2 = bbox
        color = 'g' if bbox in bbox_list[:num_true] else 'r'
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor=color, facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.show()

def get_proportion(box, binary_map):
    y1, y2, x1, x2 = np.array(box,dtype=np.int16)
    num_true = np.sum(binary_map[y1:y2, x1:x2])
    num_tot = np.abs((y2 - y1) * (x2 - x1))

    return num_true / num_tot

def get_subimages(
        label_file_path, image_file_path,
        num_true=100, num_false=100, true_thresh=0.5, pos_rand=0.01, fake_rand=0.5,
        output_dir='double_check_dataset'):

    subimage_list = []
    bbox_list = []
    
    proportion_list = []
    
    """步骤0：加载"""
    label = sio.loadmat(label_file_path)
    image = Image.open(image_file_path)
    inst_map = label['inst_map']
    binary_map = inst_map != 0
    bboxs = label['bbox']
    Height, Width = inst_map.shape
    visualize_bboxes(image_file_path,bboxs)
    """步骤1：获取正例"""
    index = np.linspace(0,len(bboxs)-1,len(bboxs), dtype=np.int32)
    np.random.shuffle(index)
    for i in range(num_true):
        bbox = bboxs[index[i]]
        box_width = bbox[3] - bbox[2]
        box_height = bbox[1] - bbox[0]
        noisy_bbox = [
            bbox[0] + int(np.random.uniform(-pos_rand,pos_rand) * box_height),
            bbox[1] + int(np.random.uniform(-pos_rand,pos_rand) * box_height),
            bbox[2] + int(np.random.uniform(-pos_rand,pos_rand) * box_width),
            bbox[3] + int(np.random.uniform(-pos_rand,pos_rand) * box_width), 
        ]
        subimage = image.crop((noisy_bbox[2],noisy_bbox[0],noisy_bbox[3],noisy_bbox[1]))
        subimage_list.append(subimage)
        proportion = get_proportion(noisy_bbox, binary_map)
        proportion_list.append(proportion)
        bbox_list.append(noisy_bbox)

    """步骤2：获取反例"""
    mean_height = np.mean([bbox[1] - bbox[0] for bbox in bboxs])
    mean_width = np.mean([bbox[3] - bbox[2] for bbox in bboxs])
    while len(subimage_list) < num_false + num_true:
        box_height = np.random.uniform(1 - fake_rand, 1 + fake_rand) * mean_height
        box_width = np.random.uniform(1 - fake_rand, 1 + fake_rand) * mean_width
        y0 = np.random.uniform(0, Height - box_height -1)
        x0 = np.random.uniform(0, Width - box_width -1)
        fake_bbox = [ y0, y0 + box_height, x0, x0 + box_width]
        proportion = get_proportion(fake_bbox, binary_map)
        if proportion < true_thresh:
            subimage = image.crop((fake_bbox[2],fake_bbox[0],fake_bbox[3],fake_bbox[1]))
            subimage_list.append(subimage)
            proportion_list.append(proportion)
            bbox_list.append(fake_bbox)

    return subimage_list, proportion_list, bbox_list

# 调用get_subimages函数并获取返回值
subimage_list, proportion_list, bbox_list = get_subimages("dataset\Lizard_Labels\Labels\consep_1.mat", "dataset\Lizard_Images\consep_1.png",num_true=num_true)

# 可视化
visualize_bboxes("dataset\Lizard_Images\consep_1.png", bbox_list)