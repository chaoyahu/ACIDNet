
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from acidnet.data.util import *
from torchvision import transforms as t

    
class LOLDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __getitem__(self, index):

        folder = self.data_dir+'/low'
        folder2= self.data_dir+'/high'
        data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
        data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        num = len(data_filenames)

        im1 = load_img(data_filenames[index])
        im2 = load_img(data_filenames2[index])
        _, file1 = os.path.split(data_filenames[index])
        _, file2 = os.path.split(data_filenames2[index])
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) # make a seed with numpy generator 
        if self.transform:
            random.seed(seed) # apply this seed to img tranfsorms
            torch.manual_seed(seed) # needed for torchvision 0.7
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2) 
        return im1, im2, file1, file2

    def __len__(self):
        return 485

#原始版本
# class LOLv2DatasetFromFolder(data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         super(LOLv2DatasetFromFolder, self).__init__()
#         self.data_dir = data_dir
#         self.transform = transform

#     def __getitem__(self, index):

#         folder = self.data_dir+'/Low'
#         folder2= self.data_dir+'/Normal'
#         data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
#         data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]
        
#         im1 = load_img(data_filenames[index])
#         im2 = load_img(data_filenames2[index])
#         _, file1 = os.path.split(data_filenames[index])
#         _, file2 = os.path.split(data_filenames2[index])
#         seed = random.randint(1, 1000000)
#         seed = np.random.randint(seed) # make a seed with numpy generator 
#         if self.transform:
#             random.seed(seed) # apply this seed to img tranforms
#             torch.manual_seed(seed) # needed for torchvision 0.7
#             im1 = self.transform(im1)      
#             random.seed(seed) # apply this seed to img tranforms
#             torch.manual_seed(seed) # needed for torchvision 0.7 
#             im2 = self.transform(im2)
#         return im1, im2, file1, file2

#     def __len__(self):
#         return 685

class LOLv2DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2DatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        
        # [修复1] 在初始化时一次性加载并排序
        # 原始代码在 __getitem__ 里 listdir，会导致每个 epoch 文件顺序随机变化（实验极其不稳定）
        folder_low = join(data_dir, 'Low')
        folder_norm = join(data_dir, 'Normal')
        
        # 必须使用 sorted()，否则 Low 和 Normal 的文件可能对应不上（比如 Low[0] 是图片A，Normal[0] 却是图片B）
        self.data_filenames = sorted([join(folder_low, x) for x in listdir(folder_low) if is_image_file(x)])
        self.data_filenames2 = sorted([join(folder_norm, x) for x in listdir(folder_norm) if is_image_file(x)])

    def __getitem__(self, index):
        # [修复2] 直接通过 index 获取固定的路径
        path1 = self.data_filenames[index]
        path2 = self.data_filenames2[index]

        im1 = load_img(path1)
        im2 = load_img(path2)
        _, file1 = os.path.split(path1)
        _, file2 = os.path.split(path2)

        # [保持] 你原始的随机种子逻辑 (这是为了同步 transform)
        if self.transform:
            seed = np.random.randint(2147483647) # 这里的生成范围改大一点更安全
            
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
            
        return im1, im2, file1, file2

    def __len__(self):
        # [修复3] 动态获取长度，防止硬编码 685 导致越界或数据丢失
        return len(self.data_filenames)



# class LOLv2SynDatasetFromFolder(data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         super(LOLv2SynDatasetFromFolder, self).__init__()
#         self.data_dir = data_dir
#         self.transform = transform
#         self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#     def __getitem__(self, index):

#         folder = self.data_dir+'/Low'
#         folder2= self.data_dir+'/Normal'
#         data_filenames = [join(folder, x) for x in listdir(folder) if is_image_file(x)]
#         data_filenames2 = [join(folder2, x) for x in listdir(folder2) if is_image_file(x)]


#         im1 = load_img(data_filenames[index])
#         im2 = load_img(data_filenames2[index])
#         _, file1 = os.path.split(data_filenames[index])
#         _, file2 = os.path.split(data_filenames2[index])
#         seed = random.randint(1, 1000000)
#         seed = np.random.randint(seed) # make a seed with numpy generator 
#         if self.transform:
#             random.seed(seed) # apply this seed to img tranfsorms
#             torch.manual_seed(seed) # needed for torchvision 0.7
#             im1 = self.transform(im1)
#             random.seed(seed)
#             torch.manual_seed(seed)         
#             im2 = self.transform(im2)
#         return im1, im2, file1, file2

#     def __len__(self):
#         return 900
class LOLv2SynDatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(LOLv2SynDatasetFromFolder, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.norm = t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 【修正1】将文件读取移至初始化，防止训练极其缓慢
        # 【修正2】增加 sorted()，防止 Low 和 Normal 图片配对错乱
        folder = self.data_dir + '/Low'
        folder2 = self.data_dir + '/Normal'
        self.data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        self.data_filenames2 = sorted([join(folder2, x) for x in listdir(folder2) if is_image_file(x)])

    def __getitem__(self, index):
        # 直接使用初始化时生成好的有序列表
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        
        _, file1 = os.path.split(self.data_filenames[index])
        _, file2 = os.path.split(self.data_filenames2[index])
        
        # 保持原有的随机种子逻辑完全不变
        seed = random.randint(1, 1000000)
        seed = np.random.randint(seed) 
        if self.transform:
            random.seed(seed) 
            torch.manual_seed(seed) 
            im1 = self.transform(im1)
            random.seed(seed)
            torch.manual_seed(seed)         
            im2 = self.transform(im2)
        return im1, im2, file1, file2

    def __len__(self):
        # 【修正3】动态返回实际长度，防止越界或读取不全
        return len(self.data_filenames)



    

