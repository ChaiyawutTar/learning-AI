import os
import numpy as np
import pandas as pd

from PIL import Image
import torch
from torch.utils.data import Dataset

class CatDogMiniDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.target = pd.read_csv(image_dir + 'annotations.csv')
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        # collect data from image directory and name of img is form csv that on colum 0
        img_path = os.path.join(self.image_dir, 
                                self.target.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB")
                         , dtype=np.float32)
        # class of target in csv
        target_label = torch.tensor(int(self.target.iloc[index, 1]))
        # transform image

        if self.transform:
            image = self.transform(image=image)['image']

        return (image, target_label)


if __name__ == "__main__":

    image_dir = 'data/train/'

    dataset = CatDogMiniDataset(image_dir)

    img, target_label = dataset[0]
    print("img.shape without transform", img.shape)

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    train_transform = A.Compose([
        A.Resize(height=32, width=32),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
    ])

    dataset = CatDogMiniDataset(image_dir, transform=train_transform)

    img, target_label = dataset[0]
    print("img.shape with transform: ", img.shape)
