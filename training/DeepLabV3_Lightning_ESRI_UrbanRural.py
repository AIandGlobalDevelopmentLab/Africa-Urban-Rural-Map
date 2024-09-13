#---Imports---
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import optim, nn, utils
import lightning.pytorch as pl
import torchmetrics
from torchmetrics.classification import MulticlassConfusionMatrix, MulticlassJaccardIndex
from collections import OrderedDict


class DeepLabV3_Lightning_ESRI_UrbanRural(pl.LightningModule):
    ''' ---Wrapper class for DeepLabV3 model---
    DeepLabv3 is a fully Convolutional Neural Network (CNN) model 
    designed by a team of Google researchers to tackle the problem of semantic segmentation. 
    DeepLabv3 is an incremental update to previous (v1 & v2) DeepLab systems and easily outperforms its predecessor.'''
    
    def __init__(self, training_folds=None, validation_fold=None, test_country=None, fold_config='default'):
        super(DeepLabV3_Lightning_ESRI_UrbanRural, self).__init__()
        
        # ---model setup---
        self.core = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=8)
        self.core.backbone.conv1 = nn.Conv2d(7, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # ---statistics---
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_acc_confmat = MulticlassConfusionMatrix(num_classes=8, normalize='none', ignore_index=-1)
    
        #alternate way if input is a sequence (yearly eval)
        self.test_acc_confmat_1 = MulticlassConfusionMatrix(num_classes=8, normalize='none', ignore_index=-1)
        self.test_acc_confmat_2 = MulticlassConfusionMatrix(num_classes=8, normalize='none', ignore_index=-1)
        self.test_acc_confmat_3 = MulticlassConfusionMatrix(num_classes=8, normalize='none', ignore_index=-1)
        self.test_acc_confmat_4 = MulticlassConfusionMatrix(num_classes=8, normalize='none', ignore_index=-1)
        self.test_acc_confmat_5 = MulticlassConfusionMatrix(num_classes=8, normalize='none', ignore_index=-1)
        self.confmats = [self.test_acc_confmat_1, self.test_acc_confmat_2, self.test_acc_confmat_3, self.test_acc_confmat_4, self.test_acc_confmat_5]
        
        self.iou = MulticlassJaccardIndex(num_classes=8, average='macro', ignore_index=-1)
        
        self.epoch = 0
        self.fold_config = fold_config
        self.test_country = test_country
        
        # ---weighted loss function setup---
        if training_folds == None and validation_fold == None:
            self.loss_fun_train = nn.CrossEntropyLoss()
            self.loss_fun_val = nn.CrossEntropyLoss()
            
        else:
            train_weights = self.gather_weights(training_folds)
            self.loss_fun_train = nn.CrossEntropyLoss(weight=train_weights, ignore_index=-1)
            
            val_weights = self.gather_weights(validation_fold)
            self.loss_fun_val = nn.CrossEntropyLoss(weight=val_weights, ignore_index=-1)
        


    def gather_weights(self, fold_countries):
        
        # read csv with class counts for each country
        mimer = '/mimer/NOBACKUP/groups/globalpoverty1/albin_and_albin'
        csv_class_dist = mimer + '/other/class_distribution_country_esri.csv'
        df_country_class_count = pd.read_csv(csv_class_dist, index_col=0)
        
        # sum together class counts of all countries and normalize
        fold_class_count = np.zeros((16))
        for country in fold_countries:
            country_class_count = df_country_class_count.loc[[country]].iloc[0].to_numpy()
            fold_class_count = np.add(fold_class_count, country_class_count)  # add to total

        fold_class_dist = fold_class_count / fold_class_count.sum()

        fold_class_dist_remapped = np.zeros((8))
        fold_class_dist_remapped[0:4] = fold_class_dist[0:4]  # water, tree, flooded veg., crops
        fold_class_dist_remapped[4] = fold_class_dist[5]  # bare
        fold_class_dist_remapped[5] = fold_class_dist[8]  # rangeland
        fold_class_dist_remapped[6] = fold_class_dist[9:12].sum()  # urban
        fold_class_dist_remapped[7] = fold_class_dist[12:16].sum()  # rural

        # create loss weights
        loss_weights = 1 / np.clip(fold_class_dist_remapped, 1e-5, 1-1e-5)
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32)
        
        return loss_weights
    
    
    def forward(self, x):

        output = self.core(x)['out']

        return output
    
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    
    def training_step(self, batch, batch_idx):
        '''calculates loss for training batch'''
        images, labels = batch
        output = self.core(images)['out']
        
        train_loss = self.loss_fun_train(output, labels)
        
        self.training_step_outputs.append(train_loss)
        self.log("train_loss", train_loss, on_epoch=True, sync_dist=True, prog_bar=True)
        return train_loss
    
    
    def on_train_epoch_end(self):
        '''prints average training loss at the end of epoch'''
        outputs = self.training_step_outputs
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            loss = sum(output.mean() for output in gathered) / len(outputs)
            print(f'training loss: {loss.item()}')
        
        self.training_step_outputs.clear()
    
    
    def validation_step(self, batch, batch_idx):
        '''calculates loss for validation batch'''
        images, labels = batch
        output = self.core(images)['out']
        
        val_loss = self.loss_fun_val(output, labels)
        
        self.validation_step_outputs.append(val_loss)
        self.log('val_loss', val_loss, on_epoch=True, sync_dist=True, prog_bar=True)
    
    
    def on_validation_epoch_end(self):
        '''prints average validation loss at the end of epoch'''
        outputs = self.validation_step_outputs
        gathered = self.all_gather(outputs)
        if self.global_rank == 0:
            loss = sum(output.mean() for output in gathered) / len(outputs)
            print('---------')
            print(f'epoch {self.epoch}')
            self.epoch += 1
            print(f'validation loss: {loss.item()}')
        
        self.validation_step_outputs.clear()
    
    
    def test_step(self, batch, batch_idx):
        '''calculates accuracy for test batch'''
        """images, labels = batch
        output = self.core(images)['out']
        
        class_probs = F.softmax(output, dim=1)
        class_predictions = torch.argmax(input=class_probs, dim=1)

        # calculate pixel-wise accuracy
        num_correct_predictions = torch.eq(class_predictions, labels).sum().item()
        num_pixels = labels.numel()
        classification_acc = num_correct_predictions / num_pixels
        
        # calculate confusion matrix
        self.test_acc_confmat.update(class_predictions, labels)
        
        # update iou metric
        self.iou(class_predictions, labels)
        
        self.log('test_acc', classification_acc, on_epoch=True, sync_dist=True, prog_bar=True)
        self.log('test_IoU', self.iou, on_epoch=True, prog_bar=True)"""
        
        #alternate way if input is a sequence (yearly eval)
        
        images, labels = batch
        batch_size, seq_len, channels, height, width = images.size()
        
        output = self.core(images.view(batch_size*seq_len, channels, 250, 250))['out']
        
        class_probs = F.softmax(output, dim=1)
        class_predictions = torch.argmax(input=class_probs, dim=1)
        class_predictions = class_predictions.view(batch_size, seq_len, 250, 250)

        # calculate confusion matrix
        for i in range(5):
            self.confmats[i].update(class_predictions[:,i,:,:], labels[:,i,:,:])
        
        # update iou metric
        self.iou(class_predictions[:,i,:,:], labels[:,i,:,:])
        
        self.log('test_IoU', self.iou, on_epoch=True, prog_bar=True)
    
    
    def on_test_epoch_end(self):
        '''prints average test set confusion matrix at the end of epoch'''
        
        class_names = ['c1:Water', 'c2:Tree', 'c3:Flooded Vegetation', 'c4:Crops', 
                       'c5:Bare Ground', 'c6:Rangeland', 'c7:Rural', 
                       'c8:Urban']
        
        """confmat = self.test_acc_confmat.compute()
        df_confmat = pd.DataFrame(confmat.cpu().numpy(), index=class_names, columns=class_names)

        if self.test_country == None:
            df_confmat.to_csv(f'/mimer/NOBACKUP/groups/globalpoverty1/albin_and_albin/confusion_matrix_esri_urban_rural_{self.fold_config}.csv')
        else:
            df_confmat.to_csv(f'/mimer/NOBACKUP/groups/globalpoverty1/albin_and_albin/confusion_matrix_esri_urban_rural_{self.test_country}.csv')
        self.test_acc_confmat.reset()"""
        
        #alternate way if input is a sequence (yearly eval)
        
        for i in range(5):
            confmat = self.confmats[i].compute()
            df_confmat = pd.DataFrame(confmat.cpu().numpy(), index=class_names, columns=class_names)
            df_confmat.to_csv(f'/mimer/NOBACKUP/groups/globalpoverty1/albin_and_albin/confusion_matrix_esri_urban_rural_{self.fold_config}_2018to2022_{2018+i}.csv')
            self.confmats[i].reset()
        
    
    
    def predict_step(self, batch, batch_idx):
        '''outputs segmentation prediction'''
        images, labels = batch
        output = self.core(images)['out']
        
        class_probs = F.softmax(output, dim=1)
        class_predictions = torch.argmax(input=class_probs, dim=1)
        
        return class_predictions



    
