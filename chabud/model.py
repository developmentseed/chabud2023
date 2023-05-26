"""
ChaBuDNet model architecture code.

Code structure adapted from Lightning project seed at
https://github.com/Lightning-AI/deep-learning-project-template
"""

import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_toolbelt.losses import (
    DiceLoss,
    BinaryFocalLoss,
    BinaryLovaszLoss,
)

from chabud.tinycd_model import ChangeClassifier


class ChaBuDNet(L.LightningModule):
    """
    Neural network for performing Change detection for Burned area Delineation
    (ChaBuD) on Sentinel 2 optical satellite imagery.

    Implemented using Lightning 2.0.
    """

    def __init__(self, lr: float = 1e-3, model_name="tinycd"):
        """
        Define layers of the ChaBuDNet model.

        Parameters
        ----------
        lr : float
            The learning rate for the Adam optimizer. Default is 0.001.
        """
        super().__init__()

        # Save hyperparameters like lr, weight_decay, etc to self.hparams
        # https://pytorch-lightning.readthedocs.io/en/2.0.2/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters(logger=True)
        self.model = self._init_model(model_name)

        # Loss functions
        self.loss_bce = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(32.0), reduction="mean"
        )
        # self.loss_dice = DiceLoss(mode="binary", from_logits=True, smooth=0.1)
        # self.loss_focal = BinaryFocalLoss(alpha=0.25, gamma=2.0)

        # Evaluation metrics to know how good the segmentation results are
        self.iou = torchmetrics.JaccardIndex(task="binary", num_classes=2)

    def _init_model(self, name):
        if name == "tinycd":
            return ChangeClassifier(
                bkbn_name="efficientnet_b4",
                pretrained=True,
                output_layer_bkbn="3",
                freeze_backbone=False,
            )
        else:
            return NotImplementedError(f"model {name} is not available")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Inference/Prediction).
        """
        y_hat: torch.Tensor = self.model(x1, x2)

        return y_hat

    def shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        batch_idx: int,
        phase: str,
    ) -> torch.Tensor:
        """
        Logic for the neural network's loop.
        """
        # dtype = torch.float16 if "16" in self.trainer.precision else torch.float32
        pre_img, post_img, mask, metadata = batch
        # y_hat is logits
        y_hat: torch.Tensor = self(x1=pre_img, x2=post_img).squeeze()
        y_pred: torch.Tensor = (F.sigmoid(y_hat) > 0.5).detach().byte()

        # Log loss and metrics
        loss: torch.Tensor = self.loss_bce(y_hat, mask.float())
        metric: torch.Tensor = self.iou(preds=y_pred, target=mask)
        self.log_dict(
            dictionary={f"{phase}/loss_dice": loss, f"{phase}/iou": metric},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Logic for the neural network's training loop.
        """
        return self.shared_step(batch, batch_idx, "train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Logic for the neural network's validation loop.
        """
        return self.shared_step(batch, batch_idx, "val")

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
