from torch.utils.data import Dataset
from pathlib import Path

REPO_ROOT_PATH = Path(__file__).parents[2]
TRAIN_SET_FILE_NAME = ""
print(REPO_ROOT_PATH)

class DigitDataset(Dataset):
    def __init__(self, transform, dataset_path : Path):
        super().__init__()

        self.transform = transform
        self.dataset_path = dataset_path




    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
