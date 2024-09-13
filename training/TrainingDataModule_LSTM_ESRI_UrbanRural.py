#!/usr/bin/env python
# coding: utf-8
# %%
# ---imports---
import sys
import os
import random
sys.path.append('.')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

import webdataset as wds
from braceexpand import braceexpand
import rasterio
from rasterio import MemoryFile


# --- Class Codices ---

# -ESRI LULC 2020 (9 classes): class codec
# 1(->0): Water
# 2(->1): Tree
# 4(->2): Flooded Vegetation
# 5(->3): Crops
# 7(->6): Built Area
# 8(->4): Bare Ground
# 9(->(-1)): Snow / Ice
# 10(->(-1)): Clouds
# 11(->5): Rangeland
# 12(->(-1)): Missing
esri_classes = [1, 2, 4, 5, 7, 8, 9, 10, 11, 12]
labels = [0, 1, 2, 3, 6, 4, -1, -1, 5, -1]

esri_class_to_index_map = np.zeros(max(esri_classes) + 1, dtype='int64')
esri_class_to_index_map[esri_classes] = labels


# ---GHS-SMOD 2020 (8 classes): class codec---
# 30(->7): URBAN CENTRE GRID CELL
# 23(->7): DENSE URBAN CLUSTER GRID CELL
# 22(->7): SEMI-DENSE URBAN CLUSTER GRID CELL
# 21(->7): SUBURBAN OR PERI-URBAN GRID CELL
# 13(->6): RURAL CLUSTER GRID CELL
# 12(->6): LOW DENSITY RURAL GRID CELL
# 11(->6): VERY LOW DENSITY RURAL GRID CELL
# 10(->6): WATER GRID CELL
# NoData [-Inf] -> 7
smod_classes = [10, 11, 12, 13, 21, 22, 23, 30]
labels = [6, 6, 6, 6, 7, 7, 7, 7]

smod_class_to_index_map = np.zeros(max(smod_classes) + 1, dtype='int64')
smod_class_to_index_map[smod_classes] = labels


# --- Helper Methods ---

def preprocess_labels(smod, esri_labels):
    # prepare smod 
    smod[np.isinf(smod)] = 10
    smod = smod.astype('int64')
            
    # prepare esri
    esri_inf_mask = np.isinf(esri_labels) 
    esri_labels[esri_inf_mask] = -1
    esri_labels = esri_labels.astype('int64')
    
    # label preprocessing: remap to standard class indices using class codec
    smod = smod_class_to_index_map[smod]
    esri_labels = esri_class_to_index_map[esri_labels]
    np.putmask(esri_labels, esri_labels == 6, smod)
    
    return esri_labels

def preprocess_input(image):
    # input preprocessing: normalize input bands to range (0.0, 1.0), mask missing data
    image[0:6] = (image[0:6] - 1) / (65455 - 1)  # bands SR_B2 to SR_B7: BGR, NIR, SWIR1, SWIR2 (1, 65455)
    nl_clipped = np.clip(image[6], a_min=-1.5, a_max=193565)
    image[6] = (nl_clipped + 1.5) / (193565 + 1.5)   # band avg_rad: VIIRS (-1.5, 193565)
            
    # set missing data to zero
    inf_mask = np.isinf(image)
    image[inf_mask] = 0


def get_patches_old(src):
    '''split each (1000,1000)-supertile into 16*(250,250)-subtiles'''
    
    for sample in src:
        
        # take out image datastream and read array data
        image_y1 = sample['tif_2016']
        image_y2 = sample['tif_2017']
        image_y3 = sample['tif_2018']
        image_y4 = sample['tif_2019']
        image_y5 = sample['tif_2020']
        image_timeseries = [image_y1, image_y2, image_y3, image_y4, image_y5]
        
        # arrays to hold the data
        image_seq = np.zeros((5, 7, 1000, 1000), dtype="float32")
        
        # read input from y1-y4
        for i, image_year_i in enumerate(image_timeseries[:-1]):
            with MemoryFile(image_year_i) as memfile:
                with memfile.open() as dataset:
                    image_array = dataset.read()
                    image_seq[i] = image_array
        
        # read input+target from y5
        with MemoryFile(image_timeseries[-1]) as memfile:
            with memfile.open() as dataset:
                image_array_target_year = dataset.read()
                image_seq[4] = image_array_target_year[:7]
                target_data = image_array_target_year[7:]
        
        
        # loop through each subtile
        for sub_tile_idx in range(16):
            
            # calculate offset
            r = (sub_tile_idx // 4) * 250
            c = (sub_tile_idx % 4) * 250
            
            
            # handle input data (time_seq, channels, height, width) 
            sub_tile_seq = image_seq[:, :, r:r+250, c:c+250]
            
            for sub_tile_year in sub_tile_seq:
                preprocess_input(sub_tile_year)
            
            
            # handle labels
            sub_tile = target_data[:, r:r+250, c:c+250]
            smod = sub_tile[0]
            esri_labels = sub_tile[2]
            
            labels = preprocess_labels(smod, esri_labels)
            target_seq = np.empty((5, 250, 250))
            target_seq[0:4] = -1  # first four years lack target
            target_seq[4] = labels
            
            yield (sub_tile_seq, target_seq.astype('int64'))


def get_patches(src):
    '''split each (1000,1000)-supertile into 16*(250,250)-subtiles'''
    
    for sample in src:
        
        # take out image datastream and read array data
        image_y1 = sample['tif_2018']
        labels_y1 = sample['tif_2018_labels']
        image_y2 = sample['tif_2019']
        labels_y2 = sample['tif_2019_labels']
        image_y3 = sample['tif_2020']
        image_y4 = sample['tif_2021']
        image_y5 = sample['tif_2022']
        image_timeseries = [(image_y1, labels_y1), (image_y2, labels_y2), image_y3, image_y4, image_y5]
        
        # arrays to hold the data
        image_seq = np.zeros((5, 7, 1000, 1000), dtype="float32")
        target_seq = np.zeros((5, 1000, 1000))
        
        # read input and targets from y1-y5
        for i, image_year_i in enumerate(image_timeseries):
            
            try:
                # read y1 & y2 (separate files for input and target)
                if i <= 1:
                    # input
                    with MemoryFile(image_year_i[0]) as memfile:
                        with memfile.open() as dataset:
                            image_array = dataset.read()
                            image_seq[i] = image_array
                    # target
                    with MemoryFile(image_year_i[1]) as memfile:
                        with memfile.open() as dataset:
                            image_array = dataset.read()
                            target_seq[i] = image_array
                # read y3 (input + target + smod)
                elif i == 2:
                    with MemoryFile(image_year_i) as memfile:
                        with memfile.open() as dataset:
                            image_array = dataset.read()
                            image_seq[i] = image_array[:7]
                            smod = image_array[7]
                            target_seq[i] = image_array[9]
                else: # read y4 & y5 (input + target)
                    with MemoryFile(image_year_i) as memfile:
                        with memfile.open() as dataset:
                            image_array = dataset.read()
                            image_seq[i] = image_array[:7]
                            target_seq[i] = image_array[7]
            except:
                print(f'{sample["__url__"]} contains corrupt tile')
            
        # preprocess input
        for image_y in image_seq:
                preprocess_input(image_y)
        
        # preprocess labels
        target_seq = preprocess_labels(smod, target_seq)
        
        # loop through each subtile
        for sub_tile_idx in range(16):
            
            # calculate offset
            r = (sub_tile_idx // 4) * 250
            c = (sub_tile_idx % 4) * 250
            
            # handle input data (time_seq, channels, height, width) 
            sub_tile_image_seq = image_seq[:, :, r:r+250, c:c+250]
            
            # handle labels
            sub_tile_target_seq = target_seq[:, r:r+250, c:c+250]
            
            yield (sub_tile_image_seq, sub_tile_target_seq)


def nodesplitter(src, group=None):
    '''splits shards across gpu:s and among dataloader workers'''
    
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        
        # get idx of current GPU and total number of GPUs
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        
        # get idx of current worker and total number of workers
        w_info = torch.utils.data.get_worker_info()
        worker_idx = w_info.id
        num_workers = w_info.num_workers
        
        # alternate shards between GPUs
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                # alternate shards between workers
                if count % num_workers == worker_idx:
                    #print(f"nodesplitter: rank={rank} size={size} worker={worker_idx} shard={item}", flush=True)
                    yield item
                count += 1
        #print(f"nodesplitter: rank={rank} size={size} count={count} DONE", flush=True)
    else:
        yield from src


def workersplitter(src, group=None):
    '''splits shards across gpu:s and among dataloader workers'''
    
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        
       # get idx of current worker and total number of workers
        w_info = torch.utils.data.get_worker_info()
        worker_idx = w_info.id
        num_workers = w_info.num_workers
        
        # alternate shards between workers
        for i, item in enumerate(src):
            if i  % num_workers == worker_idx:
                yield item
        #print(f"nodesplitter: rank={rank} size={size} count={count} DONE", flush=True)
    else:
        yield from src



# --- DataModule ---
class TrainingDataModule_LSTM_ESRI(pl.LightningDataModule):
    
    def __init__(self, training_countries: list, validation_countries: list = None, test_countries: list = None):
        super().__init__()
        
        self.training_countries = training_countries
        self.validation_countries = validation_countries
        self.test_countries = test_countries
        
        self.data_dir = '/mimer/NOBACKUP/groups/globalpoverty1/albin_and_albin/training_data_multiyear_2018to2022'
        self.shuffle_seed = 0
        self.num_gpus = 4
    
    
    def gather_shards(self, countries):
        '''gather list of shard paths from list of countries'''
        shards = []
        for country in countries:
            country_shard_dir = f'{self.data_dir}/{country}_Shards'
            num_shards = len(os.listdir(country_shard_dir))
            
            start_shard = '000000'
            end_shard = f'{num_shards-2:06}'  # for now remove last shard as it may be incomplete, therefore -2, not -1

            shards_path = country_shard_dir + '/shard_{' + start_shard + '..' + end_shard + '}.tar'
            country_shards = list(braceexpand(shards_path))
            shards.extend(country_shards)

        return shards
    
    
    def setup(self, stage: str):
        if stage == 'fit':
            # gather training and validation shard lists
            self.training_shards = self.gather_shards(self.training_countries)
            self.validation_shards = self.gather_shards(self.validation_countries)
        
        if stage == 'test':
            # gather test shard list
            self.test_shards = self.gather_shards(self.test_countries)
        
        if stage == 'predict':
            self.test_shards = self.gather_shards(self.test_countries)
        
    
    def build_datapipeline(self, shard_list, batch_size=128):
        '''build the webdataset pipeline, of type IterableDataset, return dataloader'''
        
        num_abundant_shards = len(shard_list) % self.num_gpus  # shards to leave out, for even gpu distribution
        shards = shard_list if num_abundant_shards == 0 else shard_list[0:-num_abundant_shards]
        
        dataset = wds.DataPipeline(
            wds.SimpleShardList(shards),
            nodesplitter,
            wds.tarfile_to_samples(),
            get_patches,
            wds.batched(batch_size)
        )
        
        dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=6)
        
        return dataloader
    
    
    def train_dataloader(self):
        '''returns dataloader of training set (note: rebuild after every epoch to shuffle shards)'''
        
        # epoch shuffle
        random.Random(self.shuffle_seed).shuffle(self.training_shards)
        self.shuffle_seed += 1
        
        # build dataset and dataloader
        train_dataloader = self.build_datapipeline(self.training_shards, batch_size=64)
        
        return train_dataloader
    
    def val_dataloader(self):
        '''returns dataloader of validation set'''
        
        # build dataset and dataloader
        val_dataloader = self.build_datapipeline(self.validation_shards, batch_size=128)
        
        return val_dataloader
    
    
    def test_dataloader(self):
        '''returns dataloader of test set'''
        
        # build dataset and dataloader
        #test_dataloader = self.build_datapipeline(self.test_shards, batch_size=512)
        
        dataset = wds.DataPipeline(
            wds.SimpleShardList(self.test_shards),
            workersplitter,
            wds.tarfile_to_samples(),
            get_patches,
            wds.batched(128)
        )
        
        test_dataloader = DataLoader(dataset=dataset, batch_size=None, num_workers=6)
        
        return test_dataloader
    
    
    def predict_dataloader(self):
        '''returns dataloader of test set'''
        
        # build dataset and dataloader
        predict_dataloader = self.build_datapipeline(self.test_shards, batch_size=3)
        
        return predict_dataloader

# %%
