#!/bin/env bash
#SBATCH -A SNIC2022-3-38
#SBATCH -p alvis
#SBATCH -t 0-18:00:00
#SBATCH --gpus-per-node=A100:4
#SBATCH -C MEM512
#SBATCH -J lightning_deeplabv3_train_esri_urban_rural
#SBATCH -o deeplabv3_esri_urban_rural_%a_41_to_50

apptainer exec ../my_pytorch.sif python ../training_and_inference_scripts/Training_Lightning_ESRI_UrbanRural.py --startfold $SLURM_ARRAY_TASK_ID
