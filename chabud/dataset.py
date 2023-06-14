import os
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import lightning as L
import numpy as np
from torch.utils.data import Dataset, DataLoader


class ChaBuDDataset(Dataset):
    def __init__(self, data_dir: Path, transform=None):
        self.data_dir = data_dir
        self.uuids = list(data_dir.glob("*.npz"))
        self.transform = transform

    def __getitem__(self, idx):
        uuid = self.uuids[idx]
        event = np.load(uuid)
        pre, post, mask = (
            event["pre"].astype(np.float32),
            event["post"].astype(np.float32),
            event["mask"].astype(np.uint8),
        )

        if self.transform:
            tfmed = self.transform(
                image=pre.transpose(1, 2, 0), post=post.transpose(1, 2, 0), mask=mask
            )
            pre, post, mask = tfmed["image"], tfmed["post"], tfmed["mask"]

        return (pre, post, mask, uuid.stem)

    def __len__(self):
        return len(self.uuids)


class ChaBuDDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 16,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.trn_tfm = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    p=0.5, shift_limit=0.05, scale_limit=0.05, rotate_limit=10
                ),
                ToTensorV2(),
            ],
            additional_targets={"post": "image"},
        )
        self.val_tfm = A.Compose([ToTensorV2()], additional_targets={"post": "image"})
        self.tst_tfm = A.Compose([ToTensorV2()], additional_targets={"post": "image"})

    def setup(self, stage: str | None = None) -> None:
        self.trn_ds = ChaBuDDataset(self.data_dir / "trn", transform=self.trn_tfm)
        self.val_ds = ChaBuDDataset(self.data_dir / "val", transform=self.val_tfm)
        self.tst_ds = ChaBuDDataset(self.data_dir / "val_orig", transform=self.tst_tfm)

    def train_dataloader(self):
        return DataLoader(
            self.trn_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.tst_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
