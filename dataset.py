import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
# from albumentations.pytorch import ToTensorV2

# print("Pandas version:", pd.__version__)
# print("Torchvision version:", torchvision.__version__)
# print("Albumentations version:", A.__version__)

import albumentations as A
from albumentations.pytorch import ToTensorV2

class CatDogMiniDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.target = pd.read_csv(os.path.join(image_dir, 'annotations.csv'))
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.target.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        target_label = torch.tensor(int(self.target.iloc[index, 1]))

        if self.transform:
            image = self.transform(image=image)['image']

        return (image, target_label)

if __name__ == "__main__":
    image_dir = 'data/train/'

    # Initialize dataset without transformations
    dataset = CatDogMiniDataset(image_dir)
    img, target_label = dataset[0]
    print("img.shape without transform", img.shape)

    train_transform = A.Compose([
        A.Resize(height=32, width=32),
        A.Normalize(
            mean=[0.0, 0.0, 0.0], 
            std=[1.0, 1.0, 1.0], 
            max_pixel_value=255.0
        ),
        # ToTensorV2(),  # Convert to tensor
    ])

    # Initialize dataset with transformations
    dataset = CatDogMiniDataset(image_dir, transform=train_transform)
    img, target_label = dataset[0]
    print("img.shape with transform :", img.shape)