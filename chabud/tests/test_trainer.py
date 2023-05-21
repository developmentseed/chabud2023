"""
Tests for command-line interface to execute ChaBuDNet.

Based on advanced usage of running LightningCLI from Python at
https://lightning.ai/docs/pytorch/2.0.2/cli/lightning_cli_advanced_3.html#run-from-python
"""
import pytest

from trainer import cli_main


# %%
@pytest.mark.parametrize("subcommand", ["fit", "validate"])
def test_cli_main(subcommand):
    """
    Ensure that running `python trainer.py` works with the subcommands `fit`
    and `validate`.
    """
    cli_main(args=[subcommand, "--trainer.fast_dev_run=True"])
