import lightning as L
from lightning.pytorch.callbacks import Callback
import torch
import torch.nn.functional as F
import wandb


class LogIntermediatePredictions(Callback):
    """Visualize the model results at the end of every epoch."""

    def __init__(self, logger):
        """Instantiates with wandb-logger.
        Args:
            logger : wandb-logger instance.
        """
        super().__init__()
        self.logger = logger

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """
        Called when the validation loop ends.
        At the end of each epoch, takes the first batch from validation dataset &
        logs the model predictions to wandb-logger for humans to interpret how model evolves over time.
        """
        id2label = {0: "ok", 1: "burn"}
        log_list = []

        with torch.no_grad():
            # get the first batch from trainer
            batch = next(iter(trainer.val_dataloaders))
            pre_img, post_img, mask, metadata = batch
            batch_size = mask.shape[0]

            # Pass the image through neural network model to get predicted images
            logits: torch.Tensor = pl_module(
                x1=pre_img.to(pl_module.device), x2=post_img.to(pl_module.device)
            ).squeeze()
            y_pred: torch.Tensor = F.sigmoid(logits)
            y_pred = (y_pred > 0.5).int().detach().cpu().numpy()

            for i in range(batch_size):
                log_image = wandb.Image(
                    post_img[i].permute(1, 2, 0).detach().numpy() / 3000,
                    masks={
                        "prediction": {
                            "mask_data": mask[i].detach().cpu().numpy(),
                            "class_labels": id2label,
                        },
                        "ground_truth": {
                            "mask_data": y_pred[i],
                            "class_labels": id2label,
                        },
                    },
                )
                log_list.append(log_image)

            wandb.log({"predictions": log_list})
