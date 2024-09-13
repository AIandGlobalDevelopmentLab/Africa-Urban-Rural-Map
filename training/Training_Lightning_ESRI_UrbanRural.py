#!/usr/bin/env python
# coding: utf-8
# %%
# ---imports---
import sys
import os
import random
sys.path.append('.')
sys.path.append('..')
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint

import webdataset as wds
import braceexpand
import rasterio
from rasterio import MemoryFile

from models.DeepLabV3_Lightning_ESRI_UrbanRural import DeepLabV3_Lightning_ESRI_UrbanRural
from TrainingDataModule_ESRI_UrbanRural import TrainingDataModule_ESRI_UrbanRural



if __name__ == "__main__":   

    # CLI arguments
    parser = ArgumentParser()
    parser.add_argument("--startfold", type=int, default=1)
    args = parser.parse_args()
    
    # Step 1: Define split into training, validation and test --> Create DataModule
    fold_1 = ['Algeria', 'Niger', 'Mauritania', 'Mozambique', 'CentralAfricanRepublic', 'Zimbabwe', 'Guinea', 'Malawi', 'Togo']
    fold_2 = ['DemocraticRepublicoftheCongo', 'Angola', 'Egypt', 'Zambia', 'Madagascar', 'Congo', 'Ghana', 'Eritrea', 'Guinea-Bissau']
    fold_3 = ['Sudan', 'Mali', 'UnitedRepublicofTanzania', 'Morocco', 'WesternSahara', 'Botswana', 'CotedIvoire', 'Uganda', 'Benin', 'Lesotho']
    fold_4 = ['Libya', 'SouthAfrica', 'Nigeria', 'SouthSudan', 'Kenya', 'BurkinaFaso', 'Senegal', 'Liberia', 'EquatorialGuinea']
    fold_5 = ['Chad', 'Ethiopia', 'Namibia', 'Somalia', 'Cameroon', 'Gabon', 'Tunisia', 'SierraLeone', 'Burundi']
    folds = [fold_1, fold_2, fold_3, fold_4, fold_5]
    
    fold_order = ''
    f = args.startfold
    for _ in range(5):
        fold_order += str(f)
        f = (f%5) + 1
    print(fold_order)
    
    training_folds =  folds[int(fold_order[0]) - 1] + folds[int(fold_order[1]) - 1] + folds[int(fold_order[2]) - 1]
    val_fold = folds[int(fold_order[3]) - 1]
    test_fold = folds[int(fold_order[4]) - 1]
    
    lightning_datamodule = TrainingDataModule_ESRI_UrbanRural(training_countries = training_folds, 
                                              validation_countries = val_fold, 
                                              test_countries = test_fold)
    # Step 2: Create Model
    model_checkpoints = ['12345_latest-epoch=9-step=63000.ckpt', '23451_latest-epoch=9-step=58500.ckpt', 
                         '34512_latest-epoch=9-step=53000.ckpt', '45123_latest-epoch=9-step=56000.ckpt',
                         '51234_latest-epoch=9-step=58500.ckpt']
    
    current_ckpt = f'/mimer/NOBACKUP/groups/globalpoverty1/albin_and_albin/scripts_and_notebooks/job_scripts/lightning_logs/deeplabv3_esri_urban_rural_{fold_order}_31_to_40/checkpoints/{model_checkpoints[args.startfold-1]}'
    
    #lightning_model = DeepLabV3_Lightning_ESRI_UrbanRural(training_folds, val_fold)  # start from scratch
    lightning_model = DeepLabV3_Lightning_ESRI_UrbanRural.load_from_checkpoint(current_ckpt, training_folds=training_folds, validation_fold=val_fold)  # start from checkpoint
    
    
    # Step 3: Create Trainer --> Start training
    
    # Save the model with lowest validation loss
    val_checkpoint = ModelCheckpoint(filename=fold_order + "_{epoch}-{step}-{val_loss:.3f}",
                                     monitor="val_loss",
                                     mode="min",
                                     save_top_k=3)
    
    # Save latest model
    latest_checkpoint = ModelCheckpoint(filename=fold_order + "_latest-{epoch}-{step}",
                                        monitor="step",
                                        mode="max",
                                        every_n_train_steps=500,
                                        save_top_k=2)
    
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=4, 
                         strategy='ddp', use_distributed_sampler=False, 
                         enable_progress_bar=False, reload_dataloaders_every_n_epochs=1, 
                         sync_batchnorm=True, fast_dev_run=False, enable_model_summary=False,
                         callbacks=[latest_checkpoint, val_checkpoint])
    
    
    trainer.fit(model=lightning_model, datamodule=lightning_datamodule)


# %%
