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
        ]
    )
    datamodule.setup()

    it = iter(datamodule.train_dataloader())
    pre_image, post_image, mask, metadata = next(it)

    assert pre_image.shape == (32, 512, 512, 12)
    assert pre_image.dtype == torch.int16

    assert post_image.shape == (32, 512, 512, 12)
    assert post_image.dtype == torch.int16

    assert mask.shape == (32, 512, 512)
    assert mask.dtype == torch.uint8

    assert len(metadata) == 32
    assert set(metadata[0].keys()) == {"filename", "uuid"}
