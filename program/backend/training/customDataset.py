from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        mask = self.y[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        mask = torch.from_numpy(mask)
        mask = mask.float()
        
        if mask.ndim == 3:
            mask = mask.squeeze(-1) 
            
        mask = mask.unsqueeze(0)
        
        return image, mask