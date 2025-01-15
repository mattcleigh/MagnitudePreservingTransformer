import random

import numpy as np
import torch as T
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """Text dataset for language modelling.

    Randomly samples blocks of text. Length and getitem is simply used for
    defining epoch length.
    """

    def __init__(self, file_path: str, max_seq_len: int, epoch_size: int) -> None:
        super().__init__()
        self.epoch_size = epoch_size
        self.max_seq_len = max_seq_len
        self.from_bin = False

        if file_path.endswith(".npy"):
            self.data = T.from_numpy(np.load(file_path))
        elif file_path.endswith(".bin"):
            self.from_bin = True
            self.data = T.from_numpy(np.fromfile(file_path, dtype=np.uint16))
    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, idx: int) -> T.Tensor:
        idx = random.randint(0, len(self.data) - self.max_seq_len - 1)
        x = self.data[idx : idx + self.max_seq_len]
        y = self.data[idx + 1 : idx + 1 + self.max_seq_len]
        return x.long(), y.long()


class TextModule(LightningDataModule):
    def __init__(
        self,
        *,
        train_path: str,
        val_path: str,
        test_path: str,
        max_seq_len: int,
        train_epoch_size: int,
        val_epoch_size: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> None:
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.max_seq_len = max_seq_len
        self.train_epoch_size = train_epoch_size
        self.val_epoch_size = val_epoch_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: str) -> None:
        """Sets up the relevant datasets."""
        if stage in {"fit", "train"}:
            self.train_set = TextDataset(
                self.train_path,
                self.max_seq_len,
                self.train_epoch_size,
            )
            self.val_set = TextDataset(
                self.val_path,
                self.max_seq_len,
                self.val_epoch_size,
            )
        if stage in {"predict", "test"}:
            self.test_set = TextDataset(
                self.test_path,
                self.max_seq_len,
                self.val_epoch_size,
            )

    def get_dataloader(self, dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
        )

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.train_set)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.val_set)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.test_set)

    def predict_dataloader(self) -> DataLoader:
        return self.test_dataloader()
