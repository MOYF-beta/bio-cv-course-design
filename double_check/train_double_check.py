import os
from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np

from prefetch_dataloader import NuclearSegmentDataset, PreFetchDataLoader
np.seterr(divide='ignore',invalid='ignore')
import torch
import torch.nn as nn
from state_clasifier import SimpleNet
from get_dataset import get_subimages
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32
EPOCH = 25
LR = 0.01
SGD_MOMENTUM = 0.9
TYPE = '.'

PRELOAD = 10

class DataLoaderPrefetch(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__()) 
    
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
def eval(model, eval_loader, show_result = False):
    model.eval()
    total_loss = 0
    total_samples = 0
    criterion = torch.nn.MSELoss() 
    idx = 0
    with torch.no_grad():
        for inputs, targets in eval_loader:
            targets = targets.view(-1, 1)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        if show_result:  
            plt.ion()  # 开启交互模式
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
            plt.ioff()

    mse = total_loss / total_samples
    return mse

if __name__ == '__main__':
    import torch.optim as optim
    from tqdm import tqdm
   
    begin_time = datetime.now().strftime('%y_%m_%d_%H_%M')
    
    model_dir = f'./net_weights/{TYPE}_{begin_time}_LR{LR}_BS{BATCH_SIZE}'

    os.makedirs(model_dir)
    
    

    # 初始化网络、优化器
    net = SimpleNet().to(device=device)
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=SGD_MOMENTUM)

    eval_dataset = NuclearSegmentDataset('dataset\Lizard_Images','dataset\Lizard_Labels\Labels', num_true=128, num_false=128, device=device)
    eval_dataloader = DataLoaderPrefetch(eval_dataset, batch_size=64, shuffle=True)
    eval_mse = eval(net,eval_dataloader)
    datasets = PreFetchDataLoader('dataset\Lizard_Images','dataset\Lizard_Labels\Labels',
                                        num_true=256,num_false=256,TYPE=TYPE, device=device)

    pbar = tqdm(total=EPOCH, desc="init", position=0, leave=False, ncols=80)
    pbar.set_description(f'EPOCH> loss:x.xx,eval_MSE:{eval_mse:.2f} ')
    for epoch in range(EPOCH):
        # 加载数据集
        pbar.update(1)
        train_dataset = datasets.fetch()
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

        pbar.set_description(f'EPOCH> loss:{loss:.2f},eval_MSE:{eval_acc:.2f} ')

    torch.save(net.state_dict(), f'{model_dir}/final.pth')
    torch.save(net.state_dict(), f'./net_weights/{TYPE}_latest.pth')