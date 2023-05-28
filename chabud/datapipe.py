"""
LightningDataModule that loads directly from HDF5 using torch DataPipe.
"""
import os
from typing import Iterator

import datatree
import lightning as L
import numpy as np
import torch
import torchdata
import torchdata.dataloader2
import xarray as xr


# %%
def _path_fn(urlpath: str) -> str:
    """
    Get the filename from a urlpath and prepend it with 'data' so that it is
    like 'data/filename.hdf5'.
    """
    return os.path.join("data", os.path.basename(urlpath))


def _datatree_to_chip(hdf5_file: str) -> Iterator[xr.Dataset]:
    """
    Read a nested HDF5 file into a datatree.DataTree, and iterate over each
    group which contains a chip, to produce an xarray.Dataset output with
    dimensions (12, 512, 512).
    """
    dt: datatree.DataTree = datatree.open_datatree(
        hdf5_file, engine="h5netcdf", phony_dims="access"
    )
    # Loop through every 512x512 chip stored as groups in the DataTree
    for chip in dt.values():
        _chip: xr.Dataset = chip.squeeze().to_dataset()
        _chip.attrs["uuid"] = chip.name

        # Change from channel last to channel first
        # assert list(_chip.sizes.values()) == [512, 512, 12]  # Height, Width, Channel
        name_dict = {
            old_name: new_name
            for old_name, new_name in zip(
                _chip.dims.keys(), ("height", "width", "channels")
            )
        }
        _chip = _chip.rename(name_dict=name_dict)
        _chip = _chip.transpose("channels", "height", "width")
        # assert _chip.post_fire.shape == (12, 512, 512)
        # assert _chip.mask.shape == (512, 512)

        yield _chip


def _has_pre_post_mask(dataset: xr.Dataset) -> bool:
    """
    Filter out chips that have incomplete data variables (e.g. missing
    'pre-fire'). Return True if all of ('pre_fire', 'post-fire', 'mask')
    data_vars are in the xarray.Dataset, else return False.
    """
    return set(dataset.data_vars) == {"pre_fire", "post_fire", "mask"}


def _train_val_fold(chip: xr.Dataset) -> int:
    """
    Fold 0 is used for validation, Fold 1 and above is for training.
    See https://huggingface.co/datasets/chabud-team/chabud-ecml-pkdd2023/discussions/3
    """
    if "fold" not in chip.attrs:  # no 'fold' attribute, use for training too
        return 1  # Training set
    if chip.attrs["fold"] == 0:
        return 0  # Validation set
    elif chip.attrs["fold"] >= 1:
        return 1  # Training set


def _pre_post_mask_tuple(
    dataset: xr.Dataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    From a single xarray.Dataset, split it into a tuple containing the
    pre/post/target tensors and a dictionary object containing metadata.

    Returns
    -------
    data_tuple : tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]
        A tuple with 4 objects, the pre-event image, the post-event image, the
        mask image, and a Python dict containing metadata (e.g. filename, UUID,
        fold, comments).
    """
    # return just the RGB bands for now
    pre = dataset.pre_fire.data[[3, 2, 1], ...].astype(dtype="float32")
    post = dataset.post_fire.data[[3, 2, 1], ...].astype(dtype="float32")
    mask = dataset.mask.data.astype(dtype="uint8")

    return (
        torch.as_tensor(data=pre),
        torch.as_tensor(data=post),
        torch.as_tensor(data=mask),
        {
            "filename": os.path.basename(dataset.encoding["source"]),
            **dataset.attrs,
        },
    )


def _stack_tensor_collate_fn(
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict]]:
    """
    Stack a list of torch.Tensor objects into a single torch.Tensor, and
    combine metadata attributes into a list of dicts.
    """
    pre_tensor: torch.Tensor = torch.stack(tensors=[sample[0] for sample in samples])
    post_tensor: torch.Tensor = torch.stack(tensors=[sample[1] for sample in samples])
    mask_tensor: torch.Tensor = torch.stack(tensors=[sample[2] for sample in samples])
    metadata: list[dict] = [sample[3] for sample in samples]

    return pre_tensor, post_tensor, mask_tensor, metadata


class ChaBuDDataPipeModule(L.LightningDataModule):
    """
    Lightning DataModule for loading Hierarchical Data Format 5 (HDF5) files
    from the ChaBuD-ECML-PKDD2023 competition.

    Uses torch DataPipes.

    References:
    - https://pytorch.org/data/0.6/dp_tutorial.html
    - https://zen3geo.readthedocs.io/en/v0.6.0/stacking.html
    - https://huggingface.co/datasets/chabud-team/chabud-ecml-pkdd2023/blob/main/loader.py
    - https://gitlab.com/frontierdevelopmentlab/2022-us-sarchangedetection/deepslide/-/blob/main/src/datamodules/datapipemodule.py
    """

    def __init__(
        self,
        hdf5_urls: list[str] = [
            # From https://huggingface.co/datasets/chabud-team/chabud-ecml-pkdd2023/tree/main
            "https://huggingface.co/datasets/chabud-team/chabud-ecml-pkdd2023/resolve/main/train_eval.hdf5",
            # From https://huggingface.co/datasets/chabud-team/chabud-extra/tree/main
            "https://huggingface.co/datasets/chabud-team/chabud-extra/resolve/main/california_0.hdf5",
            "https://huggingface.co/datasets/chabud-team/chabud-extra/resolve/main/california_1.hdf5",
            "https://huggingface.co/datasets/chabud-team/chabud-extra/resolve/main/california_2.hdf5",
        ],
        batch_size: int = 8,
    ):
        """
        Go from multiple HDF5 files to 512x512 chips!

        Also does mini-batching and train/validation splits.

        Parameters
        ----------
        hdf5_urls : list[str]
            A list of URLs to HDF5 files to read from. E.g.
            ``['https://.../file1.hdf5', 'https://.../file2.hdf5']``.

        batch_size : int
            Size of each mini-batch. Default is 32.

        Returns
        -------
        datapipe : torchdata.datapipes.iter.IterDataPipe
            A torch DataPipe that can be passed into a torch DataLoader.
        """
        super().__init__()
        self.hdf5_urls: list[str] = list(hdf5_urls)
        self.batch_size: int = batch_size

    def setup(
        self, stage: str | None = None
    ) -> tuple[
        torchdata.datapipes.iter.IterDataPipe, torchdata.datapipes.iter.IterDataPipe
    ]:
        """
        Data operations to perform on every GPU.
        Split data into training and test sets, etc.

        Returns
        -------
        datapipes : tuple[IterDataPipe, IterDataPipe]
            Two torch DataPipe objects to iterate over, the training set
            datapipe and the validation set datapipe.
        """
        # Step 0 - Iterate through all the HDF5 files
        dp_urls: torchdata.datapipes.iter.IterDataPipe = (
            torchdata.datapipes.iter.IterableWrapper(iterable=self.hdf5_urls)
        )
        # Step 1 - Download and cache HDF5 files to the data/ folder
        # Also includes sha256 checksum verification
        dp_cache: torchdata.datapipes.iter.IterDataPipe = dp_urls.on_disk_cache(
            filepath_fn=_path_fn,
            hash_dict={
                "data/train_eval.hdf5": "7aaf771259e81131e08671c9ecaeb2902378530957771a35fd142b157cb09931",  # 5.88GB
                "data/california_0.hdf5": "f2036e129849263b66cdb9fd4769742c499a879f91a364c42bb5c953052787fc",  # 3.38GB
                "data/california_1.hdf5": "cdb13d720fcb3115c9e1c096e22a9d652ac122c93adfcbf271d4e3684a7679af",  # 3.7GB
                "data/california_2.hdf5": "0af569c8930348109b495a5f2768758a52a6deec85768fd70c0efd9370f84578",  # 368MB
            },
            hash_type="sha256",
        )
        dp_http: torchdata.datapipes.iter.IterDataPipe = (
            dp_cache.read_from_http().end_caching(mode="wb", same_filepath_fn=True)
        )

        # Step 2 - Read HDF5 files into a DataTree and produce 512x512x12 chips
        # Also filter out chips with missing pre-fire/post-fire/mask data_vars
        dp_chip = dp_http.flatmap(fn=_datatree_to_chip).filter(
            filter_fn=_has_pre_post_mask
        )

        # Step 3 - Split chips into train/val sets based on fold attribute
        # buffer_size=-1 means that the entire dataset is buffered in memory
        dp_val, dp_train = dp_chip.demux(
            num_instances=2, classifier_fn=_train_val_fold, buffer_size=-1
        )

        # Step 4 - Convert from xarray.Dataset to tuple of torch.Tensor objects
        # Also do batching, shuffling (for train set only) and tensor stacking
        self.datapipe_train = (
            dp_train.map(fn=_pre_post_mask_tuple)
            .batch(batch_size=self.batch_size)
            .in_batch_shuffle()
            .collate(collate_fn=_stack_tensor_collate_fn)
        )
        self.datapipe_val = (
            dp_val.map(fn=_pre_post_mask_tuple)
            .batch(batch_size=self.batch_size)
            .collate(collate_fn=_stack_tensor_collate_fn)
        )

    def train_dataloader(self) -> torchdata.dataloader2.DataLoader2:
        """
        Loads the data used in the training loop.
        """
        return torchdata.dataloader2.DataLoader2(datapipe=self.datapipe_train)

    def val_dataloader(self) -> torchdata.dataloader2.DataLoader2:
        """
        Loads the data used in the validation loop.
        """
        return torchdata.dataloader2.DataLoader2(datapipe=self.datapipe_val)

    def test_dataloader(self) -> torchdata.dataloader2.DataLoader2:
        """
        Loads the data used in the test loop.
        """
        return torchdata.dataloader2.DataLoader2(datapipe=self.datapipe_val)
