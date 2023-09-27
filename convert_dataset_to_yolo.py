import os
import threading

from convert_to_yolo import convert_label_to_yolo

def process_labels(file_list):
    for filename in file_list:
        if filename.endswith(".mat"):
            label_file_path = os.path.join(input_directory, filename)
            convert_label_to_yolo(label_file_path)

def split_file_list(file_list, num_threads):
    avg = len(file_list) // num_threads
    remainder = len(file_list) % num_threads
    split_indices = [i * avg + min(i, remainder) for i in range(num_threads + 1)]
    return [file_list[split_indices[i]:split_indices[i + 1]] for i in range(num_threads)]

# 设置要处理的目录
input_directory = 'C:\\Users\\14152\\Desktop\\nuclear-segment\\dataset\\Lizard_Labels\\Labels'

# 获取目录中所有文件的列表
file_list = os.listdir(input_directory)

# 设置线程数量
num_threads = 10

# 将文件列表均分给各个线程
file_lists_for_threads = split_file_list(file_list, num_threads)

# 创建线程来处理文件列表
threads = []

for file_list_for_thread in file_lists_for_threads:
    thread = threading.Thread(target=process_labels, args=(file_list_for_thread,))
    threads.append(thread)
    thread.start()

# 等待所有线程完成
for thread in threads:
    thread.join()
