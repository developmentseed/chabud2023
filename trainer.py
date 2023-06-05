"""
Command line interface to run the neural network model!

From the project root directory, do:

    python trainer.py fit

References:
- https://lightning.ai/docs/pytorch/2.0.2/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""
import os
import sys
from pathlib import Path


import torch
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.cli import ArgsType, LightningCLI

from chabud.datapipe import ChaBuDDataPipeModule
from chabud.model import ChaBuDNet
from chabud.callbacks import LogIntermediatePredictions
from chabud.utils import load_debugger


def main(stage: str = "train", ckpt_path: str = None):
    cwd = os.getcwd()
    (Path(cwd) / "logs").mkdir(exist_ok=True)

    name = sys.argv[1]

    # LOGGERs
    name = sys.argv[1]
    wandb_logger = WandbLogger(
        project="chabud2023",
        name=name,
        save_dir="logs",
        log_model=False,
    )
    csv_logger = CSVLogger(save_dir="logs/csv_logger", name=name)

    # CALLBACKS
    lr_cb = LearningRateMonitor(
        logging_interval="step",
        log_momentum=True,
    )
    ckpt_cb = ModelCheckpoint(
        monitor="val/iou",
        mode="max",
        save_top_k=2,
        verbose=True,
        filename="epoch:{epoch}-step:{step}-loss:{val/loss:.3f}-iou:{val/iou:.3f}",
        auto_insert_metric_name=False,
    )
    log_preds_cb = LogIntermediatePredictions(logger=wandb_logger)

    # DATAMODULE
    batch_size = 16
    dm = ChaBuDDataPipeModule(batch_size=batch_size)
    dm.setup()

    # MODEL
    model = ChaBuDNet(
        lr=1e-3,
        model_name="tinycd",
        submission_filepath=f"{name}-submission.csv",
        batch_size=batch_size,
    )

    debug = False
    trainer = L.Trainer(
        fast_dev_run=False,
        limit_train_batches=2 if debug else 1.0,
        limit_val_batches=2 if debug else 1.0,
        limit_test_batches=2 if debug else 1.0,
        devices=1,
        accelerator="gpu",
        precision="16-mixed",
        max_epochs=2 if debug else 30,
        accumulate_grad_batches=1,
        logger=[
            csv_logger,
            wandb_logger,
        ],
        callbacks=[ckpt_cb, log_preds_cb],
        log_every_n_steps=1,
    )

    if stage == "train":
        # TRAIN
        print("TRAIN")
        trainer.fit(
            model,
            train_dataloaders=dm.train_dataloader(),
            val_dataloaders=dm.val_dataloader(),
        )

    # EVAL
    device = "cuda"
    print("EVAL")
    model = ChaBuDNet.load_from_checkpoint(
        ckpt_cb.best_model_path if ckpt_path is None else ckpt_path
    ).to(device)
    model.eval()
    model.freeze()
    trainer.test(model, dataloaders=dm.test_dataloader())


def cli_main(
    save_config_callback=None,
    seed_everything_default=42,
    trainer_defaults: dict = {"logger": False},
    args: ArgsType = None,
):
    """
    Command-line inteface to run ChaBuDNet with ChaBuDDataPipeModule.
    """
    cli = LightningCLI(
        model_class=ChaBuDNet,
        datamodule_class=ChaBuDDataPipeModule,
        save_config_callback=save_config_callback,
        seed_everything_default=seed_everything_default,
        trainer_defaults=trainer_defaults,
        args=args,
    )


if __name__ == "__main__":
    # cli_main()
    main(
        stage="eval",
        ckpt_path="logs/csv_logger/tinycd-baseline-16-mixed-shuffle-dataset-with-optimized-logs/version_0/checkpoints/epoch:17-step:252-loss:0.826-iou:0.459.ckpt",
    )
    print("Done!")
