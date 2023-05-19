"""
Tests for ChaBuDNet.

Based loosely on Pytorch Lightning's testing method described at
https://github.com/Lightning-AI/lightning/blob/2.0.2/.github/CONTRIBUTING.md#how-to-add-new-tests
"""
import lightning as L
import pytest
import torch
import torchdata
import torchdata.dataloader2

from chabud.model import ChaBuDNet


# %%
@pytest.fixture(scope="function", name="datapipe")
def fixture_datapipe() -> torchdata.datapipes.iter.IterDataPipe:
    """
    A torchdata DataPipe with random data to use in the tests.
    """
    datapipe = torchdata.datapipes.iter.IterableWrapper(
        iterable=[
            (
                torch.randn(8, 12, 512, 512).to(dtype=torch.int16),  # pre_image
                torch.randn(8, 12, 512, 512).to(dtype=torch.int16),  # post_image
                torch.randint(
                    low=0, high=1, size=(8, 512, 512), dtype=torch.uint8
                ),  # mask
                [{"uuid": None} for _ in range(8)],  # metadata
            )
        ]
    )
    return datapipe


# %%
def test_model(datapipe):
    """
    Run a full train, val, test and prediction loop using 1 batch.
    """
    # Get some random data
    dataloader = torchdata.dataloader2.DataLoader2(datapipe=datapipe)

    # Initialize Model
    model: L.LightningModule = ChaBuDNet()

    # Training
    trainer: L.Trainer = L.Trainer(accelerator="auto", devices=1, fast_dev_run=True)
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Test
    # TODO
