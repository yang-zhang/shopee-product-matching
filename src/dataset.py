import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset


class ShopeeDataset(Dataset):
    def __init__(self, df, mode="train", transform=None):
        self.df = df.reset_index()
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image = cv2.imread(row.filepath)[:, :, ::-1]

        if self.transform is not None:
            res = self.transform(image=image)
            image = res["image"].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)
        if self.mode == "test":
            return torch.tensor(image)
        else:
            return torch.tensor(image), torch.tensor(row.label_group)


def get_transforms(image_size):

    transforms_train = albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ImageCompression(quality_lower=99, quality_upper=100),
            albumentations.ShiftScaleRotate(
                shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7
            ),
            albumentations.Resize(image_size, image_size),
            albumentations.Cutout(
                max_h_size=int(image_size * 0.4),
                max_w_size=int(image_size * 0.4),
                num_holes=1,
                p=0.5,
            ),
            albumentations.Normalize(),
        ]
    )

    transforms_val = albumentations.Compose(
        [albumentations.Resize(image_size, image_size), albumentations.Normalize()]
    )

    return transforms_train, transforms_val
