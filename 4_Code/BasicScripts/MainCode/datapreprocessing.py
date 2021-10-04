# -*- coding: utf-8 -*-
"""

@author: ninad k gaikwad

Module - datapreprocessing

"""

## Import Modules
import tensorflow as tf
import numpy as np


def DataPreprocessor(Architecture, batch_size, buffer_size, TestData_Size ):
    
    
    if (Architecture <= 3): # Model is GAN
    
        MNIST_Training_Images, MNIST_Testing_Images = DataPreprocessor_GAN(batch_size, buffer_size, TestData_Size)        
    
    else : # Model is VAE
    
        MNIST_Training_Images, MNIST_Testing_Images = DataPreprocessor_VAE(batch_size, buffer_size, TestData_Size)
    
    return MNIST_Training_Images, MNIST_Testing_Images

def DataPreprocessor_GAN(batch_size, buffer_size, TestData_Size):
    
    ## Getting MINST Data
    (MNIST_Training_Images1, MNIST_Training_Labels1), (MNIST_Testing_Images1, MNIST_Testing_Labels1) = tf.keras.datasets.mnist.load_data()
    
    ## Reshaping
    MNIST_Training_Images1 = MNIST_Training_Images1.reshape(MNIST_Training_Images1.shape[0], 28, 28, 1).astype('float32')
    MNIST_Testing_Images1 = MNIST_Testing_Images1.reshape(MNIST_Testing_Images1.shape[0], 28, 28, 1).astype('float32')
    
    ## Normalizing to [-1,1]
    MNIST_Training_Images1 = (MNIST_Training_Images1 - 127.5) / 127.5   
    MNIST_Testing_Images1 = (MNIST_Training_Images1 - 127.5) / 127.5 
    
    ## Buffer and Batch Training Data
    MNIST_Training_Images = tf.data.Dataset.from_tensor_slices(MNIST_Training_Images1 ).shuffle(buffer_size).batch(batch_size)     
    MNIST_Testing_Images = tf.data.Dataset.from_tensor_slices(MNIST_Testing_Images1 ).shuffle(TestData_Size).batch(batch_size) 

    return MNIST_Training_Images, MNIST_Testing_Images

def DataPreprocessor_VAE(batch_size, buffer_size, TestData_Size):
    
    ## Getting MINST Data
    (MNIST_Training_Images1, MNIST_Training_Labels1), (MNIST_Testing_Images1, MNIST_Testing_Labels1) = tf.keras.datasets.mnist.load_data()
        
    ## Normalizing to [0,1]
    MNIST_Training_Images1 = MNIST_Training_Images1.reshape((MNIST_Training_Images1.shape[0], 28, 28, 1)) / 255.
    np.where(MNIST_Training_Images1 > .5, 1.0, 0.0).astype('float32')
    
    MNIST_Testing_Images1 = MNIST_Testing_Images1.reshape((MNIST_Testing_Images1.shape[0], 28, 28, 1)) / 255.
    np.where(MNIST_Testing_Images1 > .5, 1.0, 0.0).astype('float32') 
    
    ## Shuffling and Batching
    MNIST_Training_Images = tf.data.Dataset.from_tensor_slices(MNIST_Training_Images1 ).shuffle(buffer_size).batch(batch_size)     
    MNIST_Testing_Images = tf.data.Dataset.from_tensor_slices(MNIST_Testing_Images1 ).shuffle(TestData_Size).batch(batch_size) 

    return MNIST_Training_Images, MNIST_Testing_Images

   

