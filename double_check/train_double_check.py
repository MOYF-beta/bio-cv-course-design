import os
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from state_clasifier import SimpleNet
from get_dataset import get_subimages
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCH = 10
LR = 0.01
SGD_MOMENTUM = 0.9

class NuclearSegmentDataset(Dataset):

    def __init__(self, image_path, label_path, num_true=BATCH_SIZE // 2, num_false=BATCH_SIZE // 2) -> None:
        self.label_path = label_path
        self.image_path = image_path
        self.img_list = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png')]
        self.label_list = [f for f in os.listdir(label_path) if f.endswith('.mat')]
        self.num_true = num_true
        self.num_false = num_false
        self.data = []

        # Load data in __init__
        for idx_img in range(len(self.img_list)):
            subimg_list, proportion_list = get_subimages(
                os.path.join(self.label_path, self.label_list[idx_img]),
                os.path.join(self.image_path, self.img_list[idx_img]),
                self.num_true, self.num_false
            )
            subimg_list_T = [
                torch.tensor(np.array(subimg.resize((32, 32))),dtype=torch.float, device=device).transpose(0,2)
                for subimg in subimg_list]
            proportion_list_T = [
                torch.tensor(proportion, dtype=torch.float32, device=device)
                for proportion in proportion_list]
            self.data += zip(subimg_list_T, proportion_list_T)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataLoaderPrefetch(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__()) 
    
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
def eval(model, eval_loader):
    model.eval()
    total_loss = 0
    total_samples = 0
    criterion = torch.nn.MSELoss()
    
    
    idx = 0
    
    plt.ion()  # 开启交互模式

    with torch.no_grad():
        for inputs, targets in eval_loader:
            targets = targets.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
        for i in range(20):
            rand_img = np.random.randint(0,len(inputs)-1)
            ax = axes[idx//5, idx%5]
            ax.imshow(inputs[rand_img].cpu().transpose(0,2)/255)
            ax.set_title(f'O: {outputs[rand_img].item():.2f}, T: {targets[rand_img].item():.2f}')
            ax.axis('off')
            idx += 1
            
            plt.draw()  # 刷新图窗
            plt.pause(0.001)  # 给足够的时间显示图像
    
    plt.show()

    mse = total_loss / total_samples
    return mse

if __name__ == '__main__':
    import torch.optim as optim
    from tqdm import tqdm
   
    begin_time = datetime.now().strftime('%y_%m_%d_%H_%M')
    
    model_dir = f'./net_weights/{begin_time}_LR{LR}_BS{BATCH_SIZE}'

    os.makedirs(model_dir)
    
    

    # 初始化网络、优化器
    net = SimpleNet().to(device=device)
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=SGD_MOMENTUM)

    eval_dataset = NuclearSegmentDataset('dataset\Lizard_Images','dataset\Lizard_Labels\Labels')
    eval_dataloader = DataLoaderPrefetch(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_mse = eval(net,eval_dataloader)

    pbar = tqdm(total=EPOCH, desc="init", position=0, leave=False, ncols=80)
    pbar.set_description(f'EP:({0}/{EPOCH})loss:x.xx,eval_MSE:{eval_mse:.2f}')
    for epoch in range(EPOCH):
        # 加载数据集
        pbar.update(1)
        train_dataset = NuclearSegmentDataset('dataset\Lizard_Images','dataset\Lizard_Labels\Labels')
        train_dataloader = DataLoaderPrefetch(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.L1Loss()
        
        for i, (images, ideal_val) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)
                                           , position=1, leave=False, ncols=80):

            images = images
            # 将梯度清零
            optimizer.zero_grad()

            # 前向传播计算损失
            outputs = net(images)
            ideal_val = ideal_val.view(-1, 1)
            loss = criterion(outputs, ideal_val)

            # 反向传播计算梯度并更新参数
            loss.backward()
            optimizer.step()

        eval_acc = eval(net,eval_dataloader)
        if epoch>0 and epoch % 25 == 0:
            
            torch.save(net.state_dict(), f'{model_dir}/EP{i}.pth')

        pbar.set_description(f'EP:({epoch+1}/{EPOCH})loss:{loss:.2f},eval_MSE:{eval_acc:.2f}')

    torch.save(net.state_dict(), f'{model_dir}/final.pth')
    torch.save(net.state_dict(), './net_weights/latest.pth')