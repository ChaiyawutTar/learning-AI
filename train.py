import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

from dataset import CatDogMiniDataset
from model import MyNet
from logger import Logger

def main():
    """
    1. get dataset loader
        1.1 define preprocessing tranform steps
        1.2 create Dataset object (define where to load data)
        1.3 create DataLoader object (define batchsize and how to load data)
    """
    train_transform = A.Compose(
        [
            A.Resize(height=32, width=32),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2()
        ]
    )

    train_data_object = CatDogMiniDataset(image_dir='data/train/', 
                                          transform=train_transform)
    
    train_loader = DataLoader(train_data_object,
                              batch_size=32,
                              num_workers=2,
                              pin_memory=True,
                              shuffle=True
    )

    """
    2. define model components
        2.1 network
        2.2 define loss function
        2.3 define optimizer
    """

    network = MyNet()

    if torch.cuda.is_available():
        device="cuda"
    elif torch.backends.mps.is_available():
        device="mps"
    else:
        device="cpu"

    network.to(device)

    loss_fn = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    # 3. logger object
    logger = Logger(device)

    # 4. t raning loop
    print('start training')

    for epoch in range(1000):
        for batch_idx, (x, target) in enumerate(train_loader):
            # 20k picture and pick 32 at a time per loop
            # 4.0 set device
            x =  x.to(device)
            target = target.to(device)

            # 4.1 make prediction (forward pass)
            y_pred = network(x)

            # 4.2 compute loss
            loss = loss_fn(y_pred, target)

            # 4.3 compute gradients
            optimizer.zero_grad()
            loss.backward()

            # 4.4 update weights
            optimizer.step()

            # 4.5 collect results into logger
            logger.log_step(loss.item())
            # print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")  


    
if __name__ == "__main__":
    main() 