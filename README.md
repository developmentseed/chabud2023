# ChaBuD2023

A Machine Learning data pipeline for the
[Change detection for Burned area Delineation (ChaBuD)](https://huggingface.co/spaces/competitions/ChaBuD-ECML-PKDD2023)
challenge at the [ECML/PKDD 2023](https://2023.ecmlpkdd.org/submissions/discovery-challenge/challenges)
conference.

# Getting started

### Quickstart

Launch on regular [Binder](https://mybinder.readthedocs.io/en/latest).

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/developmentseed/chabud2023/main)

## Installation

### Basic

To help out with development, start by cloning this [repo-url](/../../)

    git clone <repo-url>

Then I recommend [using mamba](https://mamba.readthedocs.io/en/latest/installation.html)
to install the dependencies.
A virtual environment will also be created with Python and
[JupyterLab](https://github.com/jupyterlab/jupyterlab) installed.

    cd chabud2023
    mamba env create --file environment.yml

Activate the virtual environment first.

    mamba activate chabud

Finally, double-check that the libraries have been installed.

    mamba list

### Advanced

This is for those who want full reproducibility of the virtual environment.
Create a virtual environment with just Python and conda-lock installed first.

    mamba create --name chabud python=3.11 conda-lock=1.4.0
    mamba activate chabud

Generate a unified [`conda-lock.yml`](https://github.com/conda/conda-lock) file
based on the dependency specification in `environment.yml`. Use only when
creating a new `conda-lock.yml` file or refreshing an existing one.

    conda-lock lock --mamba --file environment.yml

Installing/Updating a virtual environment from a lockile. Use this to sync your
dependencies to the exact versions in the `conda-lock.yml` file.

    conda-lock install --mamba --name chabud conda-lock.yml

See also https://conda.github.io/conda-lock/output/#unified-lockfile for more
usage details.

## Usage

### Running jupyter lab

    mamba activate chabud
    python -m ipykernel install --user --name chabud  # to install virtual env properly
    jupyter kernelspec list --json                    # see if kernel is installed
    jupyter lab &

### Running the model

The neural network model can be ran via
[LightningCLI v2](https://pytorch-lightning.medium.com/introducing-lightningcli-v2-supercharge-your-training-c070d43c7dd6).
To check out the different options available, and look at the hyperparameter
configurations, run:

    python trainer.py --help
    python trainer.py test --print_config

To quickly test the model on one batch in the validation set:

    python trainer.py validate --trainer.fast_dev_run=True

To train the model for a hundred epochs and log metrics to
[WandB](https://wandb.ai/devseed/chabud2023):

    python trainer.py fit --trainer.max_epochs=100 \
                          --trainer.logger=WandbLogger \
                          --trainer.logger.entity=devseed \
                          --trainer.logger.project=chabud2023

To generate the CSV file of predicted masks on the validation set for
[submission](https://huggingface.co/datasets/chabud-team/chabud-ecml-pkdd2023/blob/main/create_sample_submission.py)
to https://huggingface.co/spaces/competitions/ChaBuD-ECML-PKDD2023.

    python trainer.py test --model.submission_filepath=submission.csv

More options can be found using `python trainer.py fit --help`, or at the
[LightningCLI docs](https://lightning.ai/docs/pytorch/2.0.2/cli/lightning_cli.html).
