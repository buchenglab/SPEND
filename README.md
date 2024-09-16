# Self-Permutation Noise2Noise Denoiser (SPEND)

This repository contains the implementation of the Self-Permutation Noise2Noise Denoiser (SPEND) framework designed to remove spatially correlated and spectrally varied noise from hyperspectral images. The method utilizes a novel axis-based permutation strategy to improve denoising performance by disrupting noise correlations across different data dimensions.

# Prerequesite
The code relies on the CSBDeep Python package (https://github.com/CSBDeep/CSBDeep) for U-net denoiser implementation. A copy of the csbdeep package is included in the folder.

Software dependencies and tested versions: Python 3 (3.9.16), Tensorflow 2 (2.12.0) with GPU support (CUDA (9.1) and cuDNN (8.8.0))

CSBDeep copyright:
BSD 3-Clause License
Copyright (c) 2018, Uwe Schmidt, Martin Weigert
All rights reserved.

## Overview

Hyperspectral SRS data often suffers from correlated noise in both spatial and spectral domains, making traditional denoising methods less effective. The SPEND framework introduces a self-permutation-based Noise2Noise (N2N) denoising method. By selecting a permutation axis based on the noise correlation levels along different axes, this approach maximizes the independence of input-target image pairs, improving the performance of Noise2Noise learning.

![Video_labeled](https://github.com/user-attachments/assets/ba8d133a-a794-4153-9914-1c7e256df11c)


### Key Features
- **Permutation Axis Selection**: The axis with the least noise correlation (often the spectral ω axis) is chosen for permutation. This ensures that the noise is effectively decorrelated, improving denoising performance.
- **Data Permutation**: Raw hyperspectral data is split into odd and even slices along the selected axis, and these slices are alternately concatenated to form the input and target pairs for Noise2Noise training.
- **U-Net Architecture**: A U-Net-based convolutional neural network is employed to perform the denoising task, ensuring that both spatial and spectral features are preserved during the denoising process.

## Framework Details

The SPEND framework consists of the following main steps:

1. **Permutation Axis Selection**: Based on the noise correlation levels in the data, a suitable axis is chosen (typically the spectral axis) for the permutation process.
2. **Data Permutation**: The raw hyperspectral SRS image stack is split into odd and even slices along the chosen axis. These slices are alternately concatenated to form the input and target datasets for the Noise2Noise model.
3. **Training Phase**: A U-Net architecture is used to train on the permuted input-target pairs, leveraging the independence created by the permutation process to estimate and remove noise.
4. **Prediction Phase**: The trained model is applied to the original data, maintaining the continuity and integrity of the spatial and spectral information to produce denoised images.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/SPEND.git
    cd SPEND
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **datasplit_with_aug_choose.py** prepares your hyperspectral SRS dataset and specifies the permutation axis based on noise correlation analysis.
2. **1_datapre_SPEND.py** creates input and target datasets.
3. **2_training_SPEND.py** trains the SPEND model.
4. **3_prediction_SPEND.py** applies the trained model to perform denoising on your dataset.

## Citation

If you use this code in your work, please consider citing our paper:

