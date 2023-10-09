# PyTorch imports:
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T

# Numpy imports:
import numpy as np

# Pandas imports:
import pandas as pd

# OpenCV imports:
import cv2

# Other imports:
import os
import yaml
from pathlib import Path
from functools import partial


DEFAULT_TRANSFORMS = [partial(cv2.resize, dsize=(128, 128)),
                    partial(cv2.erode, kernel=np.ones(shape=(5, 5)), iterations=2),
                      lambda image : cv2.threshold(src=image, thresh=1, maxval=255, type=cv2.THRESH_BINARY)[1],
                      ]
DEFAULT_TRANSFORMS = None
class DigitDataset(Dataset):
    IMAGE_SIZE = (28, 28)
    DEFAULT_YAML_PATH = Path(__file__).parent / "dataset_path.yaml"

    def __init__(self, transforms : torchvision.transforms=DEFAULT_TRANSFORMS, dataset_yaml_path : str=DEFAULT_YAML_PATH, train : bool=True):
        self.dataset_yaml_path = dataset_yaml_path

        self.dataset_path = None
        if train:
            dataset_type = 'train'
        else:
            dataset_type = 'test'

        self.dataset_path = DigitDataset.load_dataset(dataset_yaml_path)[dataset_type]
        self.dataset, self.labels = DigitDataset.load_dataset_csv(self.dataset_path)
        self.transforms = transforms

    @staticmethod
    def load_dataset(path : str):
        with open(path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        return yaml_content


    @staticmethod
    def load_dataset_csv(dataset_path : str):
        dataset_df = pd.read_csv(dataset_path)
        labels = dataset_df.label
        dataset = dataset_df.drop("label", axis=1)

        return dataset, labels

    @staticmethod
    def transform_image(image : np.ndarray, transforms : list):
        for transform in transforms:
            image = transform(image)
        return image


    def __getitem__(self, idx):
        image = self.dataset.iloc[idx]
        image = cv2.convertScaleAbs(image.to_numpy().reshape(DigitDataset.IMAGE_SIZE).astype(int))
        label = self.labels[idx]
        if self.transforms is not None:
            image = DigitDataset.transform_image(image, self.transforms)
        return image, label

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":

    import random

    dset = DigitDataset()
    idx = random.randint(0, len(dset))
    image, label =  dset[idx]

    cv2.imshow(str(label), image)
    cv2.waitKey(0)
