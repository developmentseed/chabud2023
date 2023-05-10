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
Make an explicit [`conda-lock.yml`](https://github.com/conda/conda-lock) file
based on the dependency specification in `environment.yml` (only needed if
creating a new virtual environment/refreshing an existing one).

    mamba env create --file environment.yml
    mamba activate chabud
    conda-lock lock --mamba --file environment.yml

Creating/Installing a virtual environment from a conda lock file.
See also https://conda.github.io/conda-lock/output/#unified-lockfile

    mamba create --name chabud python=3.11 conda-lock=1.4.0
    mamba activate chabud
    conda-lock install --name chabud conda-lock.yml

## Usage

### Running jupyter lab

    mamba activate chabud
    python -m ipykernel install --user --name chabud  # to install virtual env properly
    jupyter kernelspec list --json                    # see if kernel is installed
    jupyter lab &
