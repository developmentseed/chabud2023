"""
Tests for ChaBuDDataPipeModule.

Integration test for the entire data pipeline from loading the data and
pre-processing steps, up to the DataLoader producing mini-batches.
"""

import lightning as L
import torch

from chabud.datapipe import ChaBuDDataPipeModule


# %%
def test_datapipemodule():
    """
    Ensure that ChaBuDDataPipeModule works to load data from a nested HDF5 file
    into torch.Tensor and list objects.
    """
    datamodule: L.LightningDataModule = ChaBuDDataPipeModule(
        hdf5_urls=[
            "https://huggingface.co/datasets/chabud-team/chabud-extra/resolve/main/california_2.hdf5"
        ],
        batch_size=8,
    )
    datamodule.setup()

    it = iter(datamodule.train_dataloader())
    pre_image, post_image, mask, metadata = next(it)

    assert pre_image.shape == (8, 3, 512, 512)
    assert pre_image.dtype == torch.float32

    assert post_image.shape == (8, 3, 512, 512)
    assert post_image.dtype == torch.float32

    assert mask.shape == (8, 512, 512)
    assert mask.dtype == torch.uint8

    assert len(metadata) == 8
    assert set(metadata[0].keys()) == {"filename", "uuid"}
