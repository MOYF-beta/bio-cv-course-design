import scipy.io as sio
import numpy as np
import os

def convert_label_to_yolo(label_file_path,output_dir = 'yolo_labels'):
    """Converts a label file to YOLO format."""
    label = sio.loadmat(label_file_path)

    inst_map = label['inst_map']
    Height, Width = inst_map.shape

    nuclei_id = label['id']
    classes = label['class']
    bboxs = label['bbox']
    centroids = label['centroid']

    unique_values = np.unique(inst_map).tolist()[1:]

    nuclei_id = np.squeeze(nuclei_id).tolist()

    os.makedirs(output_dir, exist_ok=True)

    yolo_format_lines = []
    for value in unique_values:
        idx = nuclei_id.index(value)

        class_ = classes[idx]
        bbox = bboxs[idx]
        centroid = centroids[idx]

        x_center = (bbox[2] + bbox[3]) / (2.0 * Width)
        y_center = (bbox[0] + bbox[1]) / (2.0 * Height)
        width = (bbox[3] - bbox[2]) / Width
        height = (bbox[1] - bbox[0]) / Height
        yolo_format_lines.append(f"{class_[0]} {x_center} {y_center} {width} {height}\n")

    file_name_with_extension = os.path.basename(label_file_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)

    txt_file_name = os.path.join(output_dir, f'{file_name}.txt')
    with open(txt_file_name, 'w') as f:
        f.writelines(yolo_format_lines)

#convert_label_to_yolo("C:\\Users\\14152\\Desktop\\nuclear-segment\\dataset\\Lizard_Labels\\Labels\\dpath_1.mat")