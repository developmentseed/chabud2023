"""
ChaBuDNet model architecture code.

Code structure adapted from Lightning project seed at
https://github.com/Lightning-AI/deep-learning-project-template
"""
import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import BinaryJaccardIndex
import trimesh.voxel.runlength
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import (
    FocalLoss,
    LovaszLoss,
    DiceLoss,
)

from chabud.tinycd_model import ChangeClassifier
from chabud.unet_model import UnetChangeClassifier


class ChaBuDNet(L.LightningModule):
    """
    Neural network for performing Change detection for Burned area Delineation
    (ChaBuD) on Sentinel 2 optical satellite imagery.

    Implemented using Lightning 2.0.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        model_name="tinycd",
        submission_filepath: str = "submission.csv",
    ):
        """
        Define layers of the ChaBuDNet model.

        Based on the TinyCD model with a Siamese U-Net architecture consisting
        of an EfficientNet-b4 backbone, Mix and Attention Mask Block (MAMB) and
        bottleneck mixing block, up-sampling decoder, and a pixel level
        classifier (PW-MLP). Using Pytorch implementation from
        https://github.com/AndreaCodegoni/Tiny_model_4_CD

        |      Backbone     |          'Neck'           |         Head        |
        |-------------------|---------------------------|---------------------|
        |  EfficientNet-b4  |  MAMB + Mixing bottleneck |  Upsample + PW-MLP  |

        Reference:
        - Codegoni, A., Lombardi, G., & Ferrari, A. (2022). TINYCD: A (Not So)
          Deep Learning Model For Change Detection (arXiv:2207.13159). arXiv.
          https://doi.org/10.48550/arXiv.2207.13159

        Parameters
        ----------
        lr : float
            The learning rate for the Adam optimizer. Default is 0.001.

        model_name : str
            Name of the neural network model to use. Choose from ['tinycd'].
            Default is 'tinycd'.

        submission_filepath : str
            Filepath of the CSV file to save the output used for submitting to
            HuggingFace after running the test_step. Default is
            `submission.csv`.
        """
        super().__init__()

        # Save hyperparameters like lr, weight_decay, etc to self.hparams
        # https://pytorch-lightning.readthedocs.io/en/2.0.2/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters(logger=True)
        self.model = self._init_model(model_name)

        # Loss functions
        self.criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(5.0), reduction="mean"
        )
        # self.criterion = DiceLoss(mode="binary", from_logits=True, smooth=0.1)
        # self.criterion = FocalLoss(mode="binary", alpha=0.25, gamma=2.0)

        # Evaluation metrics to know how good the segmentation results are
        self.iou = BinaryJaccardIndex(threshold=0.5)

    def _init_model(self, name):
        if name == "tinycd":
            return ChangeClassifier(
                bkbn_name="efficientnet_b4",
                pretrained=True,
                output_layer_bkbn="3",
                freeze_backbone=False,
            )
        elif name == "unet":
            return UnetChangeClassifier()
        else:
            return NotImplementedError(f"model {name} is not available")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Inference/Prediction).
        """
        if self.hparams.model_name == "tinycd":
            y_hat: torch.Tensor = self.model(x1, x2)
        elif self.hparams.model_name == "unet":
            y_hat: torch.Tensor = self.model(x1, x2)
        else:
            raise NotImplementedError(
                f"model {self.hparams.model_name} is not available"
            )

        return y_hat

    def shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        batch_idx: int,
        phase: str,
        log: bool = True,
    ) -> torch.Tensor:
        """
        Logic for the neural network's loop.
        """
        # dtype = torch.float16 if "16" in self.trainer.precision else torch.float32
        pre_img, post_img, mask, metadata = batch
        logits: torch.Tensor = self(x1=pre_img, x2=post_img).squeeze()
        y_pred: torch.Tensor = F.sigmoid(logits).detach()

        # Compute loss and metrics
        loss: torch.Tensor = self.criterion(logits, mask.float())
        metric: torch.Tensor = self.iou(y_pred, mask)

        if log:
            self._log(loss, metric, phase)

        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Logic for the neural network's training loop.
        """
        return self.shared_step(batch, batch_idx, phase="train", log=True)

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Logic for the neural network's validation loop.
        """
        return self.shared_step(batch, batch_idx, phase="val", log=True)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Logic for the neural network's test loop.

        Produces a CSV table for submission. Columns are 'id', 'index', and
        'rle_mask', where 'rle_mask' is a binary run length encoding of the
        burned area mask.

        References:
        - https://huggingface.co/datasets/chabud-team/chabud-ecml-pkdd2023/blob/main/create_sample_submission.py
        - https://trimsh.org/trimesh.voxel.runlength.html#trimesh.voxel.runlength.dense_to_brle
        """
        pre_img, post_img, mask, metadata = batch

        # Pass the image through neural network model to get predicted images
        logits: torch.Tensor = self(x1=pre_img, x2=post_img).squeeze()
        y_pred: torch.Tensor = F.sigmoid(logits).detach()

        # Format predicted mask as binary run length encoding vector
        result: list = []
        for pred_mask, uuid in zip(y_pred, map(lambda x: x["uuid"], metadata)):
            flat_binary_mask: np.ndarray = (y_pred > 0.5).cpu().flatten().numpy()
            brle: np.ndarray = trimesh.voxel.runlength.dense_to_brle(
                dense_data=flat_binary_mask
            )
            encoded_prediction: dict = {
                "id": uuid,
                "rle_mask": brle,
                "index": np.arange(len(brle)),
            }
            result.append(pd.DataFrame(data=encoded_prediction))

        # Write results to CSV. Create file on first batch, append afterwards
        df_submission: pd.DataFrame = pd.concat(objs=result)
        df_submission.to_csv(
            path_or_buf=self.hparams.submission_filepath,
            index=False,
            mode="w" if batch_idx == 0 else "a",
            header=True if batch_idx == 0 else False,
        )

        # Log loss and metric
        loss: torch.Tensor = self.criterion(logits, mask.float())
        metric: torch.Tensor = self.iou(y_pred, mask)
        self._log(loss, metric, "test")

        return metric

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Optimizing function used to reduce the loss, so that the predicted
        mask gets as close as possible to the groundtruth mask.

        Using the Adam optimizer with a learning rate of 0.001. See:

        - Kingma, D. P., & Ba, J. (2017). Adam: A Method for Stochastic
          Optimization. ArXiv:1412.6980 [Cs]. http://arxiv.org/abs/1412.6980

        Documentation at:
        https://lightning.ai/docs/pytorch/2.0.2/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)

    def _log(self, loss, metric, phase):
        on_step = True if phase == "train" else False

        self.log(
            f"{phase}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{phase}/iou",
            metric,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
