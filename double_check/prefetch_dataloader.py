import os
import threading
import queue
import numpy as np

import torch
from get_dataset import get_subimages
from torch.utils.data import  Dataset

class NuclearSegmentDataset(Dataset):

    def __init__(self, image_path, label_path, num_true=32, num_false=32, TYPE=".", device = "cpu") -> None:
        self.label_path = label_path
        self.image_path = image_path
        self.img_list = [f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.png') and (TYPE in f)]
        self.label_list = [f for f in os.listdir(label_path) if f.endswith('.mat') and (TYPE in f)]
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



# 初始化预加载器
class PreFetchDataLoader:
    def __init__(self, image_path, label_path, num_true=32, num_false=32, TYPE=".", device = "cpu", preload = 2):

        self.image_path = image_path
        self.label_path = label_path
        self.num_true = num_true
        self.num_false = num_false
        self.TYPE = TYPE
        self.device = device

        self.ds_list = []
        self.load_lock = threading.Lock()
        self.fetch_lock = threading.Lock()
        self.preload = preload
        self.queue = queue.Queue()

        threading.Thread(target=self.load_data, daemon=True).start()

    # 加载数据集的线程函数
    def load_data(self):
        while True:     
            with self.load_lock:
                if self.queue.qsize() < self.preload:
                    train_dataset = NuclearSegmentDataset(self.image_path, self.label_path,
                                                  num_true=self.num_true, num_false=self.num_false
                                                  , device = self.device, TYPE=self.TYPE)
                    self.queue.put(train_dataset)
                    

    # 获取数据集
    def fetch(self):
        with self.fetch_lock:
            
            train_dataset = self.queue.get()
            self.queue.task_done()

            return train_dataset