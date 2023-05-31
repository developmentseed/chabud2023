"""
Tests for ChaBuDNet.

Based loosely on Lightning's testing method described at
https://github.com/Lightning-AI/lightning/blob/2.0.2/.github/CONTRIBUTING.md#how-to-add-new-tests
"""
import os
import tempfile

import lightning as L
import pandas as pd
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
                torch.randn(2, 3, 512, 512).to(dtype=torch.float32),  # pre_image
                torch.randn(2, 3, 512, 512).to(dtype=torch.float32),  # post_image
                torch.randint(
                    low=0, high=1, size=(2, 512, 512), dtype=torch.uint8
                ),  # mask
                [{"uuid": None} for _ in range(2)],  # metadata
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

    # Test/Evaluation
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmpfile:
        trainer.model.hparams.submission_filepath = tmpfile.name
        trainer.test(model=model, dataloaders=dataloader)

        assert os.path.exists(tmpfile.name)
        df: pd.DataFrame = pd.read_csv(tmpfile.name)
        assert len(df) > 0
        assert df.columns.to_list() == ["id", "rle_mask", "index"]
