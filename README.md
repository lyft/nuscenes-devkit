# NuScenes Devkit - Lyft Level 5 AV Fork

Welcome to the devkit for the [Lyft Level 5 AV dataset](https://level5.lyft.com/dataset/)! This devkit shall help you to visualise and explore our dataset. 

## Release Notes
This devkit is a slightly modified version of the [nuScenes devkit](https://www.nuscenes.org). We added the following features:

* support for coloured rasterised maps (instead of just binary masks)
* render vehicle-centered top-down views with map in the background

Some files from the original distribution which aren't directly related to the devkit are not part of this repository.

## Getting Started

### Devkit Setup

If you have a Python >=3.6 environment set up and Pip installed, you can simply run
```
pip install -r setup/requirements.txt
```
to install the required packages. Then you're set up to use the devkit.

For more detailed installation instructions, see `./setup/installation.md`

### Dataset Download
Go to <https://level5.lyft.com/dataset/> to download the Lyft Level 5 AV Dataset.

### Tutorial
To get started with the nuScenes devkit, run the tutorial using [Jupyter Notebook](https://jupyter.org/):

   ```jupyter notebook python-sdk/tutorial_lyft.ipynb```
