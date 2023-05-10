"""
LightningDataModule that loads directly from HDF5 using torch DataPipe.
"""
import os
from typing import Optional

import datatree
import lightning as L
import torchdata
import torchdata.dataloader2
import xarray as xr


# %%
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
        batch_size: int = 32,
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
        self, stage: Optional[str] = None
    ) -> (torchdata.datapipes.iter.IterDataPipe, torchdata.datapipes.iter.IterDataPipe):
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
        def _path_fn(path) -> str:
            # print(f"Downloading from {path}")
            return os.path.join("data", os.path.basename(path))

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
        def _datatree_to_chip(hdf5_file) -> xr.Dataset:
            """
            Read a nested HDF5 file into a datatree.DataTree, and iterate
            over each group which contains a chip of dimensions (512, 512, 12),
            to produce an xarray.Dataset output with
            """
            dt = datatree.open_datatree(
                hdf5_file, engine="h5netcdf", phony_dims="access"
            )
            for chip in dt.values():
                _chip = chip.squeeze().to_dataset()
                # assert list(_chip.sizes.values()) == [512, 512, 12]  # Height, Width, Channel
                # assert set(_chip.data_vars) == {"mask", "post_fire"}
                yield _chip

        dp_chip = dp_http.flatmap(fn=_datatree_to_chip)

        # Step 3 - Split chips into train/val sets based on fold attribute
        def _train_val_fold(chip):
            """
            Fold 0 is used for validation, Fold 1 and above is for training.
            See https://huggingface.co/datasets/chabud-team/chabud-ecml-pkdd2023/discussions/3
            """
            if chip.attrs["fold"] == 0:
                return 0  # Validation set
            elif chip.attrs["fold"] >= 1:
                return 1  # Training set

        dp_train, dp_val = dp_chip.demux(
            num_instances=2, classifier_fn=_train_val_fold, buffer_size=self.batch_size
        )

        # TODO convert from xarray.Dataset to torch.Tensor

        self.datapipe_train = dp_train
        self.datapipe_val = dp_val

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
