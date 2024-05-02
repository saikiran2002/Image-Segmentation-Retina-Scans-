import torch.nn.functional as F
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2
from PIL import Image
import torchsummary
import torch.optim as optim
import numpy as np
import torchvision.utils
device = torch.device('cuda')    
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None,windows_size=(128,128),stride=(32,32)):
        self.root_dir = root_dir
        self.transform = transform
        self.window_size = windows_size
        self.stride = stride
        
        self.image_dir = os.path.join(root_dir,'images')
        self.mask_dir = os.path.join(root_dir,'1st_manual')
        self.eye_mask_dir = os.path.join(root_dir,'mask')
        
        self.image_filenames = os.listdir(self.image_dir)

    # Number of images in the folder
    def __len__(self):
        return len(self.image_filenames)

    # Getting the images
    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        image_path = os.path.join(self.image_dir,image_name)
        image = Image.open(image_path).convert('L')
        patches_image = self.extract_patches(image)

        # Extracting the patches and loading training and manual highlighted blood vessels
        mask_name = image_name.replace("training.tif","manual1.gif")
        mask_path = os.path.join(self.mask_dir,mask_name)
        mask = Image.open(mask_path).convert('L')
        patches_mask = self.extract_patches(mask)
        
        # Loading the masks
        eye_mask_name = image_name.replace(".tif","_mask.gif")
        eye_mask_path = os.path.join(self.eye_mask_dir,eye_mask_name)
        eye_mask = Image.open(eye_mask_path).convert('L')
        patches_eye_mask = self.extract_patches(eye_mask)

        data = []

        # Doing transformations
        if self.transform:
            for patch1,patch2,patch3 in zip(patches_image,patches_mask,patches_eye_mask):
                data.append((self.transform(patch1),self.transform(patch2),self.transform(patch3)))

        return data

    # Extracting patches based on Sliding window
    def extract_patches(self, image):
        patches = []
        width, height = image.size
        window_height, window_width = self.window_size
        stride_vertical, stride_horizontal = self.stride

        for y in range(0, height - window_height + 1, stride_vertical):
            for x in range(0, width - window_width + 1, stride_horizontal):
                patch = image.crop((x, y, x + window_width, y + window_height))
                patches.append(patch)

        return patches
