# -*- coding: utf-8 -*-
"""

@author: ninad k gaikwad

GAN and VAE on MNIST Data - Main File

"""
############################ Initial SET UP ##################################

## Importing Modules
import datapreprocessing
import ganbuilder 
import vaebuilder 
import trainingplotter 
import time
import tensorflow as tf


## Choose Architecture
# DenseGAN = 1
# DenseGAN_Deep = 2
# CNNGAN = 3
# DenseVAE= 4
# CNNVAE= 5

Architecture = 3

## Setting up Hyperparameters - Data
Buffer_Size = 60000 # Total = 60000
Batch_Size = 128
TestData_Size = 1000 # Total = 10000

## Setting up HyperParameters - Training
Epochs = 50
LearningRate=1e-4

## Setting up HyperParameters - GAN
GAN_Noise_Dimension = 100

## Setting up HyperParameters - VAE
VAE_LaentVar_Dimension = 2



############################ Data Preprocessing ###############################

TrainingData, TestingData = datapreprocessing.DataPreprocessor(Architecture, Batch_Size, Buffer_Size, TestData_Size )


############################## Model Creation #################################

if (Architecture <= 3): # Model is GAN

    Generator_Model  , Discriminator_Model = ganbuilder.GANModel_Create(Architecture, GAN_Noise_Dimension)        

else : # Model is VAE

    Decoder_Model  , Encoder_Model = vaebuilder.VAEModel_Create(Architecture, GAN_Noise_Dimension)
        
    
############################ Test Data Creation ###############################
 
if (Architecture <= 3): # Model is GAN

    Test_Sample= tf.random.normal([16, GAN_Noise_Dimension])      

else : # Model is VAE

    for Test_Batch in Test_Sample.take(1):
    
        Test_Sample = Test_Batch[0:16, :, :, :]
        
        
######################### Initializing Optimizers #############################
 
if (Architecture <= 3): # Model is GAN

    # Generator Optimizer
    Generator_Optimizer = tf.keras.optimizers.Adam(LearningRate) 
    
    # Discriminator Optimizer
    Discriminator_Optimizer = tf.keras.optimizers.Adam(LearningRate)    

else : # Model is VAE

    VAE_Optimizer = tf.keras.optimizers.Adam(LearningRate)
    

############################## Training Model #################################

if (Architecture <= 3): # Model is GAN

    epoch_num = 0
    
    epoch_store = []
    
    gen_loss_store = []
      
    disc_loss_store = [] 
      
    time_epoch_store = []  
    
    for epoch in range(Epochs):
        
        epoch_num = epoch_num+1
        
        epoch_store.append(epoch_num)
        
        start_time = time.time()        
        
        for Image_Batch in TrainingData:
            
            gen_loss, disc_loss, Generator_Model1, Discriminator_Model1 = ganbuilder.GAN_Training_Step(Generator_Model, Discriminator_Model, Generator_Optimizer, Discriminator_Optimizer, Batch_Size, GAN_Noise_Dimension, Image_Batch)
            
            Generator_Model =  Generator_Model1 
            
            Discriminator_Model = Discriminator_Model1 
          
        gen_loss_store.append(gen_loss.numpy())
        
        disc_loss_store.append(disc_loss.numpy())  
        
        trainingplotter.Plot_Training_GAN(Generator_Model, epoch + 1, Test_Sample, gen_loss_store, disc_loss_store, epoch_store)
        
        end_time = time.time()
        
        time_epoch = end_time - start_time 
        
        time_epoch_store.append(time_epoch)
        
        time_epochs_total = sum(time_epoch_store)     
        
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time_epoch))
        print ('Total time = {} sec'.format(time_epochs_total))
        print ('Gen Loss = {} ; Disc Loss {} '.format(gen_loss, disc_loss))   

else : # Model is VAE

    epoch_num = 0
    
    epoch_store = []
    
    loss_store = []
      
    time_epoch_store = [] 
    
    for epoch in range(Epochs ):
        
        epoch_num = epoch_num+1
          
        epoch_store.append(epoch_num)    
          
        start_time = time.time()
        
        for Image_Batch in TrainingData:
            
            VAE_Loss, Encoder_Model1, Decoder_Model1= vaebuilder.VAE_Training_Step(Encoder_Model, Decoder_Model, VAE_Optimizer, Image_Batch )
            
            Decoder_Model =  Decoder_Model1 
            
            Encoder_Model = Encoder_Model1             
        
        end_time = time.time()  
          
        loss_store.append(VAE_Loss.numpy())        
          
        time_epoch = end_time - start_time 
          
        time_epoch_store.append(time_epoch)
          
        time_epochs_total = sum(time_epoch_store)     
    
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time_epoch))
        print ('Total time = {} sec'.format(time_epochs_total))
        print ('VAE Loss = {}  '.format(VAE_Loss)) 
 

      

