# Africa-Rural-Urban-Map
This repository contains the replication code for the paper *Mapping Africa’s Settlements: High Resolution Urban and Rural Map by Deep Learning and Satellite Imagery*. It includes a link to access the urban-rural map product, instructions for using the trained model to predict LULC in any desired location, and step-by-step guidance for replicating and reproducing the study's results.

[Link to the paper](https://arxiv.org/abs/2411.02935)

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

The definition of the DeepLabV3 model used in this work is available in the [python script](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/b61ecf559a21df5db840dfada224eac73e184ddd/training/DeepLabV3_Lightning_ESRI_UrbanRural.py). 

The modules used during the training and testing phases, including the data loader, data pipeline, preprocessing, and GPU workload distribution, can be found in this [Python script](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/abaf2c8a0985eaf6504f5fa6b7a6db11032e566f/training/TrainingDataModule_ESRI_UrbanRural.py)

For the definition of the five data folds and the relevant trained models, as well as using the modules for training model, refer to this [Python script](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/abaf2c8a0985eaf6504f5fa6b7a6db11032e566f/training/Training_Lightning_ESRI_UrbanRural.py)

The model was trained using the NAISS cloud computing service by executing this [Bash script](https://github.com/AIandGlobalDevelopmentLab/LULC-Rural-Urban/blob/7da77dd2b99a1e4ea74a1324a4d42ce1b7bbbb4e/training/lightning_deeplabv3_train_esri_urban_rural.sh)

## Phase 3: Evaluation
Evaluating the performance of the model temporally [python script](https://github.com/AIandGlobalDevelopmentLab/Africa-Rural-Urban-Map/blob/f61a0086302931e9da7783edf5e3051c244b4fb5/evaluation/Testing_Lightning_ESRI_UrbanRural_2018to2022.py)

Evaluating the performance of the model country-wise [python script](https://github.com/AIandGlobalDevelopmentLab/Africa-Rural-Urban-Map/blob/f61a0086302931e9da7783edf5e3051c244b4fb5/evaluation/Testing_Lightning_ESRI_UrbanRural_CountryWise.py)

Metric analysis including accuracy, precission, recall, IoU and F1-score. [Jupyter notebook](https://github.com/AIandGlobalDevelopmentLab/Africa-Rural-Urban-Map/blob/56bdf6d0dc3be89b541096ca3e7e8ee7c0b4d148/evaluation/Metric%20Analysis%20-%20Landcover%20Prediction_ESRI_UrbanRural_Africa.ipynb)

Metric analysis for country-wise Box plot visualization [Jupyter notebook](https://github.com/AIandGlobalDevelopmentLab/Africa-Rural-Urban-Map/blob/21f08e5f511faa1e2f23fd46ba1a1015ea94b851/evaluation/Metric%20Analysis%20-%20Landcover%20Prediction_ESRI_UrbanRural_Country_Boxplot.ipynb)

Metric analysis for country-wise CSV generation [Jupyter notebook](https://github.com/AIandGlobalDevelopmentLab/Africa-Rural-Urban-Map/blob/21f08e5f511faa1e2f23fd46ba1a1015ea94b851/evaluation/Metric%20Analysis%20-%20Landcover%20Prediction_ESRI_UrbanRural_Country_ToCSV.ipynb)

Using DHS dataset to evaluate JRC SMOD and ours generated maps [GEE JavaScript](https://github.com/AIandGlobalDevelopmentLab/Africa-Rural-Urban-Map/blob/206c36b43a442cb355e57f00a06ea636e0fce02f/evaluation/DHS_evaluate.js)

## Phase 4: Prediction
Predict and visualize the LULC with rural/urban classes, and smooth the prediction using a sliding window [Jupyter notebook](https://github.com/AIandGlobalDevelopmentLab/Africa-Rural-Urban-Map/blob/046b9daa327ef4703df9e7445d51ff0816150f46/prediction/Visualize_Prediction_ESRI_UrbanRural_Smoothened_plot.ipynb)

Predict and save the LULC with rural/urban classes, and smooth the prediction using a sliding window [Jupyter notebook](https://github.com/AIandGlobalDevelopmentLab/Africa-Rural-Urban-Map/blob/046b9daa327ef4703df9e7445d51ff0816150f46/prediction/Visualize_Prediction_ESRI_UrbanRural_Smoothened_save.ipynb)

