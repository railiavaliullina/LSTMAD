import torch

from datasets.ecg5000 import ECG5000
from datasets.ecg5000 import DatasetType
import config as cfg


def get_dataloaders():

    train_dataset = ECG5000(dataset_type=DatasetType.TRAIN)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    valid_dataset = ECG5000(dataset_type=DatasetType.VALID)
    valid_dl = torch.utils.data.DataLoader(valid_dataset, batch_size=cfg.batch_size, drop_last=True)

    return train_dl, valid_dl
