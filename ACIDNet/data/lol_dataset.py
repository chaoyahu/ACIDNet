
import os
import random
import torch
import torch.utils.data as data
import numpy as np
from os import listdir
from os.path import join
from ACIDNet.data.util import *
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

# Original version
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
        
        # Load and sort file lists once during initialization.
        # Calling listdir inside __getitem__ would reshuffle files every epoch.
        folder_low = join(data_dir, 'Low')
        folder_norm = join(data_dir, 'Normal')
        
        # sorted() is required so Low and Normal stay aligned by index.
        self.data_filenames = sorted([join(folder_low, x) for x in listdir(folder_low) if is_image_file(x)])
        self.data_filenames2 = sorted([join(folder_norm, x) for x in listdir(folder_norm) if is_image_file(x)])

    def __getitem__(self, index):
        # Use the indexed paths directly.
        path1 = self.data_filenames[index]
        path2 = self.data_filenames2[index]

        im1 = load_img(path1)
        im2 = load_img(path2)
        _, file1 = os.path.split(path1)
        _, file2 = os.path.split(path2)

        # Preserve the original random seeding logic for synchronized transforms.
        if self.transform:
            seed = np.random.randint(2147483647)  # A wider range is safer here.
            
            random.seed(seed)
            torch.manual_seed(seed)
            im1 = self.transform(im1)
            
            random.seed(seed)
            torch.manual_seed(seed)
            im2 = self.transform(im2)
            
        return im1, im2, file1, file2

    def __len__(self):
        # Return the dynamic dataset length instead of a hard-coded value.
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

        # Read file lists during initialization to avoid slow training.
        # Use sorted() so Low and Normal image pairs stay aligned.
        folder = self.data_dir + '/Low'
        folder2 = self.data_dir + '/Normal'
        self.data_filenames = sorted([join(folder, x) for x in listdir(folder) if is_image_file(x)])
        self.data_filenames2 = sorted([join(folder2, x) for x in listdir(folder2) if is_image_file(x)])

    def __getitem__(self, index):
        # Reuse the ordered lists built during initialization.
        im1 = load_img(self.data_filenames[index])
        im2 = load_img(self.data_filenames2[index])
        
        _, file1 = os.path.split(self.data_filenames[index])
        _, file2 = os.path.split(self.data_filenames2[index])
        
        # Keep the original random seeding logic unchanged.
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
        # Return the actual dataset length to avoid out-of-range access.
        return len(self.data_filenames)



    
