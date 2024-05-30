# PyTorch Image Classification Project

This repository contains the modules for the PyTorch Image Classification Project.

## Overview

The PyTorch Image Classification Project is a deep learning project that aims to classify images using the PyTorch framework and is based on a TinyVGG architecture from the CNN explainer website(https://poloclub.github.io/cnn-explainer/). This repository specifically contains the modules that are used in the project.

## TinyVGG Architecture

![TinyVGG Architecture](tinyvgg_architecture.png)

## Modules

- `data_download.py`: This module downloads the data from a given url.
- `data_setup.py`: This module creates PyTorch DataLoaders from the downloaded dataset. 
- `TinyVGG_model_builder.py`: This module defines the TinyVGG architecture of the classification model.
- `engine.py`: This module contains the functions for training and testing a PyTorch model.
- `train.py`: This module is responsible for training the classification model using the provided dataset.
- `utils.py`: This module contains a utility function for saving the state dict of a trained PyTorch model.
- `predict.py`: This module uses the trained model to predict the class of an image.

## Getting Started

To get started with the PyTorch Image Classification Project, follow these steps:

1. Clone this repository to your local machine.
2. Install the required dependencies by running `pip install -r requirements.txt`.
3. Run the `train.py` script to train the classification model.
4. Use the `predict.py` script to predict the class of an image.
