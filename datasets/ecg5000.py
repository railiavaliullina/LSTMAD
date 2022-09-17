import torch
from torch.utils.data import Dataset
from pathlib import Path
import pickle
from enum import Enum, auto

import config


class DatasetType(Enum):
    TRAIN = auto()
    VALID = auto()
    VALID_MLE = auto()
    TEST = auto()


class ECG5000(Dataset):
    r"""
    http://www.timeseriesclassification.com/description.php?Dataset=ECG5000
    """

    def __init__(self, dataset_type: DatasetType, data_path: Path = config.DATA_PATH):

        if dataset_type is DatasetType.TRAIN:
            with open(data_path / 'train_0', 'rb') as f:
                self.vectors = torch.from_numpy(pickle.load(f)).float()
                self.labels = torch.zeros((self.vectors.size(0),), dtype=torch.int)
        elif dataset_type is DatasetType.VALID:
            with open(data_path / 'validation1_0', 'rb') as f:
                self.vectors = torch.from_numpy(pickle.load(f)).float()
                self.labels = torch.zeros((self.vectors.size(0),), dtype=torch.int)
        elif dataset_type is DatasetType.VALID_MLE:
            with open(data_path / 'validation2_0', 'rb') as f:
                vectors0 = torch.from_numpy(pickle.load(f)).float()
                labels0 = torch.zeros((vectors0.size(0),), dtype=torch.int)
            with open(data_path / 'validation_1', 'rb') as f:
                vectors1 = torch.from_numpy(pickle.load(f)).float()
                labels1 = torch.ones((vectors1.size(0),), dtype=torch.int)

            self.vectors = torch.cat([vectors0, vectors1], axis=0)
            self.labels = torch.cat([labels0, labels1], axis=0)
        elif dataset_type is DatasetType.TEST:
            with open(data_path / 'test_0', 'rb') as f:
                vectors0 = torch.from_numpy(pickle.load(f)).float()
                labels0 = torch.zeros((vectors0.size(0),), dtype=torch.int)
            with open(data_path / 'test_1', 'rb') as f:
                vectors1 = torch.from_numpy(pickle.load(f)).float()
                labels1 = torch.ones((vectors1.size(0),), dtype=torch.int)

            self.vectors = torch.cat([vectors0, vectors1], axis=0)
            self.labels = torch.cat([labels0, labels1], axis=0)
        else:
            raise Exception(f'Unknown dataset type "{dataset_type}"')

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.vectors[idx], self.labels[idx]
