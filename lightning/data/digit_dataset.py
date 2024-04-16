from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch
import cv2
from typing import Union

REPO_ROOT_PATH = Path(__file__).absolute().parent
DATASET_PATH_YAML_FILE_NAME = "dataset_path.yaml"
DATASET_YAML_FILE_NAME = REPO_ROOT_PATH / DATASET_PATH_YAML_FILE_NAME

class DigitDataset(Dataset):
    def __init__(self, transform, dataset_type : str='train', dataset_path_yaml : Path = DATASET_YAML_FILE_NAME):
        super().__init__()

        assert dataset_type.lower() in ['train', 'test']

        self.transform = transform
        self.dataset_path_yaml = dataset_path_yaml
        self.dataset_type = dataset_type


    def get_dataset_path(self,):
        with open(self.dataset_path_yaml, 'r') as file:
            dataset_path = yaml.safe_load(file)[self.dataset_type]
        return dataset_path


    def __getitem__(self, idx : int) -> tuple[Union[np.ndarray, torch.tensor], int]:
        with open(self.get_dataset_path()) as f:
            data_point = np.array([line for line_num, line in enumerate(f) if line_num == idx + 1][0].split(','), dtype=np.uint8)

        label, image = data_point[0], data_point[1:]
        image = image.reshape(int(np.sqrt(image.shape[0])), int(np.sqrt(image.shape[0])))

        if self.transform is None:
            return image, label
        return self.transform(image), label

    def __len__(self):
        return len(pd.read_csv(self.get_dataset_path()))


dataset = DigitDataset(transform=None)

image, label = dataset[5]

cv2.imshow(f"Digit: {label}", image)
cv2.waitKey(0)
