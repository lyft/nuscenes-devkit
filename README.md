# Lyft Dataset SDK

Welcome to the devkit for the [Lyft Level 5 AV dataset](https://level5.lyft.com/dataset/)! This devkit shall help you to visualise and explore our dataset.


## Release Notes
This devkit is based on a version of the [nuScenes devkit](https://www.nuscenes.org).

## Getting Started

### Installation

You can use pip to install [lyft-dataset-sdk](https://pypi.org/project/lyft-dataset-sdk/):
```bash
pip install -U lyft_dataset_sdk
```

If you want to get the latest version of the code before it is released on PyPI you can install the library from GitHub:

```bash
pip install -U git+https://github.com/lyft/nuscenes-devkit
```

### Dataset Download
Go to <https://level5.lyft.com/dataset/> to download the Lyft Level 5 AV Dataset.


The dataset is also availible as a part of the [Lyft 3D Object Detection for Autonomous Vehicles Challenge](https://www.kaggle.com/c/3d-object-detection-for-autonomous-vehicles).

### Tutorial and Reference Model
Check out the [tutorial and reference model README](notebooks/README.md).

![](notebooks/media/001.gif)


# Dataset structure

The dataset contains of json files:

1. `scene.json` - 25-45 seconds snippet of a car's journey.
2. `sample.json` - An annotated snapshot of a scene at a particular timestamp.
3. `sample_data.json` - Data collected from a particular sensor.
4. `sample_annotation.json` - An annotated instance of an object within our interest.
5. `instance.json` - Enumeration of all object instance we observed.
6. `category.json` - Taxonomy of object categories (e.g. vehicle, human).
7. `attribute.json` - Property of an instance that can change while the category remains the same.
8. `visibility.json` - (currently not used)
9. `sensor.json` - A specific sensor type.
10. `calibrated_sensor.json` - Definition of a particular sensor as calibrated on a particular vehicle.
11. `ego_pose.json` - Ego vehicle poses at a particular timestamp.
12. `log.json` - Log information from which the data was extracted.
13. `map.json` - Map data that is stored as binary semantic masks from a top-down view.


With [the schema](schema.md).

# Data Exploration Tutorial

To get started with the Lyft Dataset SDK, run the tutorial using [Jupyter Notebook](notebooks/tutorial_lyft.ipynb).