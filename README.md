<h1 align="center">Unified Cross-Domain Fusion and Iterative Refinement for Camouflaged Object Detection</h1>


## News :newspaper:
* **`Mar 11, 2025`:** First release.

## Overview  
This repository provides a PyTorch implementation of **IR-CDFNet**, an **Iterative Refinement Cross-Domain Fusion Network** for **Camouflaged Object Detection** (COD). IR-CDFNet integrates spatial and frequency-domain features to enhance segmentation while preserving fine details. It introduces **Domain Correlation Fusion (DCF)** for complementary feature enhancement, **Domain Difference Convolution (DDC)** for refining spatial representations, and **Iterative Refinement Masking (IRM)** for progressively restoring intricate details. Extensive experiments on four COD benchmarks demonstrate its state-of-the-art performance.

## Usage

### Installation
To use this repository, follow the steps below to set up the environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lining118/IR-CDFNet.git
   cd IR-CDFNet

2.	**Install the required dependencies:**
    It is recommended to create a virtual environment first, then install the dependencies.
    ```bash
    # PyTorch==2.0.1 is used for faster training with compilation.
    conda create -n sanet python=3.10 -y && conda activate sanet
    pip install -r requirements.txt

3. **Download the datasets:**
   [Download datasets](https://drive.google.com/drive/folders/1ehBdZcQWRVshFxR2u7-E1Uv-fwhkdOiE?usp=drive_link)
    After setting up the environment, you can download the training and test datasets from the provided links. Once downloaded, please unzip the datasets into the data folder under the root directory of the project. The folder structure should look like this:
   ```bash
   SANet/
   ├── datasets/
   │   ├── CAMO_TestingDataset/
   │   ├── CHAMELEON_TestingDataset/
   │   ├── COD10K_TestingDataset/
   │   ├── NC4K_TestingDataset/
   │   └── COD10K_CAMO_CHAMELEON_TrainingDataset/
   ├── requirements.txt
   └── … (other project files)
4. **Download the weights:**
   [Download weights](https://drive.google.com/drive/folders/1ehBdZcQWRVshFxR2u7-E1Uv-fwhkdOiE?usp=drive_link)
   After downloading the datasets, you will need to download the pretrained weights for the model and the backbone. These weights are required to initialize the model for training and inference.

   - **Model Weights:** The model weights should be placed in the `ckpt/COD` directory.
   - **Backbone Weights:** The backbone weights should be placed in the `lib/weights/backbones` directory.

### Run
```shell
# Train & Test & Evaluation
./sub.sh RUN_NAME GPU_NUMBERS_FOR_TRAINING GPU_NUMBERS_FOR_TEST
# Example: ./sub.sh  0,1,2,3,4,5,6,7 0

# See train.sh / test.sh for only training / test-evaluation.
# After the evaluation, run `gen_best_ep.py` to select the best ckpt from a specific metric (you choose it from Sm, wFm).
```




   
