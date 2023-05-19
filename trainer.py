"""
Command line interface to run the neural network model!

From the project root directory, do:

    python trainer.py fit

References:
- https://lightning.ai/docs/pytorch/2.0.2/cli/lightning_cli.html
- https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6
"""
import torch
from lightning.pytorch.cli import LightningCLI

from chabud.datapipe import ChaBuDDataPipeModule
from chabud.model import ChaBuDNet


# %%
def cli_main():
    cli = LightningCLI(
        model_class=ChaBuDNet,
        datamodule_class=ChaBuDDataPipeModule,
        save_config_callback=None,
        seed_everything_default=42,
        trainer_defaults=dict(logger=False, precision="bf16-mixed"),
    )
    print("Done!")


if __name__ == "__main__":
    cli_main()
