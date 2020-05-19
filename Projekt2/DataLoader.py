import torch
from torch.utils.data import Dataset
import numpy as np

class MyDataset(Dataset):
   
    def __init__(self, img_file, mask_file, transform=None):
       
        self.images = np.load(img_file)
        self.masks = np.load(mask_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_img = self.images[idx,:]
        sample_masks = self.masks[idx,:]
        
        if self.transform:
            sample = self.transform((sample_img, sample_masks))

        return (sample)