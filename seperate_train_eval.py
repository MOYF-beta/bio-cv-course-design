import os
import random
import shutil

# 设置数据文件夹路径
source_folder = 'C:\\Users\\14152\\Desktop\\nuclear-segment\\dataset\\Lizard_Images'
target_folder = './lizard_yolo'

# 创建训练集和测试集文件夹
train_folder = os.path.join(target_folder, 'dataset', 'train')
eval_folder = os.path.join(target_folder, 'dataset', 'eval')
os.makedirs(train_folder, exist_ok=True)
os.makedirs(eval_folder, exist_ok=True)

# 获取所有样本的文件名
samples = [f for f in os.listdir(source_folder) if f.endswith('.png')]

# 计算划分比例
train_percentage = 0.9
num_train_samples = int(len(samples) * train_percentage)
num_eval_samples = len(samples) - num_train_samples

# 随机选择样本
random.shuffle(samples)
train_samples = samples[:num_train_samples]
eval_samples = samples[num_train_samples:]

# 将训练集样本复制到相应文件夹
for sample in train_samples:
    sample_name = os.path.splitext(sample)[0]
    img_path = os.path.join(source_folder, sample_name + '.png')
    label_path = os.path.join(source_folder, sample_name + '.txt')
    shutil.copy(img_path, train_folder)
    shutil.copy(label_path, train_folder)

# 将测试集样本复制到相应文件夹
for sample in eval_samples:
    sample_name = os.path.splitext(sample)[0]
    img_path = os.path.join(source_folder, sample_name + '.png')
    label_path = os.path.join(source_folder, sample_name + '.txt')
    shutil.copy(img_path, eval_folder)
    shutil.copy(label_path, eval_folder)
