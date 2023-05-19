# ChaBuD Python modules

This folder contains Python scripts used to pre-process satellite data, as well
as the neural network model architecture, data loaders, and training/testing
scripts. To ensure high standards of reproducibility, the code is structured
using the [Lightning](https://lightning.ai/pytorch-lightning) framework and
based on https://github.com/Lightning-AI/deep-learning-project-template.

- :bricks: datapipe.py - Data pipeline to load Sentinel-2 optical imagery from HDF5 files and perform pre-processing
- :spider_web: model.py - Code containing Neural Network model architecture
