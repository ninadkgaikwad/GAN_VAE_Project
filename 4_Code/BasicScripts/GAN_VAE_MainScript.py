# -*- coding: utf-8 -*-
"""

@author: ninad k gaikwad

GAN and VAE on MNIST Data - Main File

"""
############################ Initial SET UP ##################################

## Importing Modules
import datapreprocessing
import GAN_MNIST_BasicScript_1
import VAE

## Choose Architecture
# DenseGAN = 1
# DenseGAN_Deep = 2
# CNNGAN = 3
# DenseVAE= 4
# CNNVAE= 5

Architecture = 1

## Setting up Hyperparameters - Data
TrainData_size = 60000 # Total = 60000
Batch_size = 128
TestData_size = 1000 # Total = 10000

## Setting up HyperParameters - Training
Epochs = 50
LearningRate=1e-4

## Setting up HyperParameters - VAE
VAE_LaentVar_Dimension = 2

## Setting up HyperParameters - GAN
GAN_Noise_Dimension = 2

############################ Model Training ##################################

if (Architecture <= 3): # Model is GAN

    if (Architecture == 1): # Model is DenseGAN
    
    elif (Architecture == 2): # Model is DenseGAN_Deep
    
    elif (Architecture == 3): # Model is CNNGAN
    

else : # Model is VAE

    if (Architecture == 4): # Model is DenseVAE
    
    elif (Architecture == 5): # Model is CNNVAE
    
