"""
ChaBuDNet model architecture code.

Code structure adapted from Lightning project seed at
https://github.com/Lightning-AI/deep-learning-project-template
"""
import lightning as L
import numpy as np
import pandas as pd
import torch
import torchmetrics
import trimesh.voxel.runlength


# %%
class ChaBuDNet(L.LightningModule):
    """
    Neural network for performing Change detection for Burned area Delineation
    (ChaBuD) on Sentinel 2 optical satellite imagery.

    Implemented using Lightning 2.0.
    """

    def __init__(self, lr: float = 0.001, submission_filepath: str = "submission.csv"):
        """
        Define layers of the ChaBuDNet model.

        Parameters
        ----------
        lr : float
            The learning rate for the Adam optimizer. Default is 0.001.

        submission_filepath : str
            Filepath of the CSV file to save the output used for submitting to
            HuggingFace after running the test_step. Default is
            `submission.csv`.
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
        dtype = torch.float16 if "16" in self.trainer.precision else torch.float32
        pre_img, post_img, mask, metadata = batch

        # Pass the image through neural network model to get predicted images
        x: torch.Tensor = torch.concat(tensors=[pre_img, post_img], dim=1).to(
            dtype=dtype
        )
        # assert x.shape == (32, 24, 512, 512)
        y_hat: torch.Tensor = self(x=x)
        # assert y_hat.shape == mask.shape == (32, 512, 512)

        # Format predicted mask as binary run length encoding vector
        result: list = []
        for pred_mask, uuid in zip(y_hat, pd.DataFrame(metadata)["uuid"]):
            flat_binary_mask: np.ndarray = (
                torch.sigmoid(input=pred_mask).to(bool).flatten().cpu().numpy()
            )
            brle: np.ndarray = trimesh.voxel.runlength.dense_to_brle(
                dense_data=flat_binary_mask
            )
            encoded_prediction: dict = {
                "id": uuid,
                "rle_mask": brle,
                "index": torch.arange(len(brle)),
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

        # Log validation loss and metrics
        metric: torch.Tensor = self.iou(preds=y_hat, target=mask)
        self.log_dict(
            dictionary={"test/iou": metric}, on_step=True, on_epoch=False, prog_bar=True
        )
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
