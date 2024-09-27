# LULC-Rural-Urban
This repository contains the replication code for the paper *LULC with Urban/Rural Classes in Africa Using Deep Learning and Satellite Imagery*. It includes a link to access the LULC product, instructions for using the trained model to predict LULC in any desired location, and step-by-step guidance for replicating and reproducing the study's results.

## Introduction
This README provides comprehensive instructions for setting up and using our system to generate LULC-Rural-Urban maps. It is designed to help users replicate the process by providing Python script files. The README is organized into a step-by-step guide, with important notes Furthermore, all script files are thoroughly commented to aid users in following the instructions effectively.

## Procedure Overview
The simulation setup consists of several phases, each designed to serve a specific purpose, including dataset handling, training of deep models, evaluation, and LULC prediction/map generation. The procedure can be summarized as follows:

1. Handling dataset: This phase involves generating and managing the dataset across the entire African continent.
2. Training deep models: This phase is dedicated to training the deep learning models using a five-fold dataset configuration.
3. Evaluation: This phase involves evaluating the performance of the trained models and analyzing the results.
4. Prediction: This phase focuses on using the trained models for map generation. It can be run independently for those who wish to utilize the trained model for predictions at custom locations without replicating the entire workflow.
   
Now, let's dive into each phase in detail.

## Phase 1: Handling dataset
This phase involves downloading data from GEE and preparing the dataset for both the model training and prediction phases.

Investigating the class distribution when combining LULC classes with JRC SMOD classes. [Jupyter notebook](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/bc7d844c41b1a4a569965480858dc83bfe8f9baf/data_handling/class_dist_esri_full_smod.ipynb)

Bulk Country-Wise Data Downloading from GEE: Landsat-8 and VIIRS Nighttime Light Dataset. [Jupyter notebook](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/9a7d934f811bd5b36301862bd0ec04d3d66c906b/data_handling/data_loading_inference.ipynb) We are looking to perform bulk data downloads on a country-wise basis from GEE. This will include both Landsat-8 imagery and VIIRS nighttime light datasets.

Downloading data relevant to a specific set of coordinates. [Jupyter notebook](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/99aaebfd492fe72bf8c755a2deb8ba232722d615/data_handling/data_loading_single_tile_from_coords.ipynb)

Downloading tiles covering a square area. [Jupyter notebook](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/3be2801b47f078cee1c6b9e2dd4d564d436a672e/data_handling/data_loading_single_tiles_from_list_2013_2022.ipynb) . We are looking to download data tiles that cover a defined square area.


## Phase 2: Training deep models

The definition of the utilized DeepLabV3 model that is used in this work in available in the [python script](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/b61ecf559a21df5db840dfada224eac73e184ddd/training/DeepLabV3_Lightning_ESRI_UrbanRural.py). 

Modules that are used in train/test phases. Data loader, data pipeline, pre-preocessing the data, sharing work between GPUs, etc. [Python script](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/abaf2c8a0985eaf6504f5fa6b7a6db11032e566f/training/TrainingDataModule_ESRI_UrbanRural.py)

Defining five folds and the relevent trained models. Utilizing the moduls to train the model. [Python script](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/abaf2c8a0985eaf6504f5fa6b7a6db11032e566f/training/Training_Lightning_ESRI_UrbanRural.py)

We trained the model using NAISS cloud computation service by running a [Bash script](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/7da77dd2b99a1e4ea74a1324a4d42ce1b7bbbb4e/training/lightning_deeplabv3_train_esri_urban_rural.sh)

## Phase 3: Evaluation


## Phase 4: Prediction

