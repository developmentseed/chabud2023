"""
ChaBuDNet model architecture code.

Code structure adapted from Lightning project seed at
https://github.com/Lightning-AI/deep-learning-project-template
"""

import lightning as L
import torch
import torchmetrics


# %%
class ChaBuDNet(L.LightningModule):
    """
    Neural network for performing Change detection for Burned area Delineation
    (ChaBuD) on Sentinel 2 optical satellite imagery.

    Implemented using Lightning 2.0.
    """

    def __init__(self, lr: float = 0.001):
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

        # First input convolution with same number of groups as input channels
        self.input_conv = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=24, out_channels=64, kernel_size=3, padding=1, groups=4
            ),
            torch.nn.SiLU(),
        )

        # TODO Get backbone architecture from some model zoo
        # self.backbone = ()

        # Output layers
        self.output_conv = torch.nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=3, padding=1, groups=1
        )

        # Loss functions
        self.loss_bce = torch.nn.BCEWithLogitsLoss(reduction="mean")

        # Evaluation metrics to know how good the segmentation results are
        self.iou = torchmetrics.JaccardIndex(task="binary", num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (Inference/Prediction).
        """
        x_: torch.Tensor = self.input_conv(x)
        # x_: torch.Tensor = self.backbone(x_)
        y_hat: torch.Tensor = self.output_conv(x_)

        return y_hat.squeeze()

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Logic for the neural network's training loop.
        """
        dtype = torch.float16 if "16" in self.trainer.precision else torch.float32
        pre_img, post_img, mask, metadata = batch

        # Pass the image through neural network model to get predicted images
        x: torch.Tensor = torch.concat(tensors=[pre_img, post_img], dim=1).to(
            dtype=dtype
        )
        # assert x.shape == (32, 24, 512, 512)
        y_hat: torch.Tensor = self(x=x)
        # assert y_hat.shape == mask.shape == (32, 512, 512)

        # Log training loss and metrics
        loss: torch.Tensor = self.loss_bce(input=y_hat, target=mask.to(dtype=dtype))
        metric: torch.Tensor = self.iou(preds=y_hat, target=mask)
        self.log_dict(
            dictionary={"train/loss_bce": loss, "train/iou": metric},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict],
        batch_idx: int,
    ) -> torch.Tensor:
        """
        Logic for the neural network's validation loop.
        """
        dtype = torch.float16 if "16" in self.trainer.precision else torch.float32
        pre_img, post_img, mask, metadata = batch

        # Pass the image through neural network model to get predicted images
        x: torch.Tensor = torch.concat(tensors=[pre_img, post_img], dim=1).to(
            dtype=dtype
        )
        # assert x.shape == (32, 24, 512, 512)
        y_hat: torch.Tensor = self(x=x)
        # assert y_hat.shape == mask.shape == (32, 512, 512)

        # Log validation loss and metrics
        loss: torch.Tensor = self.loss_bce(input=y_hat, target=mask.to(dtype=dtype))
        metric: torch.Tensor = self.iou(preds=y_hat, target=mask)
        self.log_dict(
            dictionary={"val/loss_bce": loss, "val/iou": metric},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return loss

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
