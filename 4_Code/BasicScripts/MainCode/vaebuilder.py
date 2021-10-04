# -*- coding: utf-8 -*-
"""

@author: ninad k gaikwad

Module - vaebuilder

"""

# Import Modules
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def VAEModel_Create(Architecture, LatentVar_Dimension):  

    if (Architecture == 4): # Model is DenseVAE
    
        Encoder_Model = tf.keras.Sequential()
        
        Encoder_Model.add(layers.Reshape((784,), input_shape=[28, 28, 1]))    
        
        Encoder_Model.add(layers.Dense(600))
        Encoder_Model.add(layers.BatchNormalization())
        Encoder_Model.add(layers.LeakyReLU())
        
        Encoder_Model.add(layers.Dense(300))
        Encoder_Model.add(layers.BatchNormalization())
        Encoder_Model.add(layers.LeakyReLU())    
    
        Encoder_Model.add(layers.Dense(100))
        Encoder_Model.add(layers.BatchNormalization())
        Encoder_Model.add(layers.LeakyReLU())
        
        Encoder_Model.add(layers.Dense(50))
        Encoder_Model.add(layers.BatchNormalization())
        Encoder_Model.add(layers.LeakyReLU())    
    
        Encoder_Model.add(layers.Dense(LatentVar_Dimension+LatentVar_Dimension))  
        
        Decoder_Model= tf.keras.Sequential()
        
        Decoder_Model.add(layers.Dense(50, input_shape=[LatentVar_Dimension]))
        Decoder_Model.add(layers.BatchNormalization())
        Decoder_Model.add(layers.LeakyReLU())
        
        Decoder_Model.add(layers.Dense(100))
        Decoder_Model.add(layers.BatchNormalization())
        Decoder_Model.add(layers.LeakyReLU())    
    
        Decoder_Model.add(layers.Dense(300))
        Decoder_Model.add(layers.BatchNormalization())
        Decoder_Model.add(layers.LeakyReLU())
        
        Decoder_Model.add(layers.Dense(600))
        Decoder_Model.add(layers.BatchNormalization())
        Decoder_Model.add(layers.LeakyReLU())     
        
        Decoder_Model.add(layers.Dense(784))
        Decoder_Model.add(layers.BatchNormalization())
        Decoder_Model.add(layers.LeakyReLU())  
    
        Decoder_Model.add(layers.Reshape((28, 28, 1)))
        assert Decoder_Model.output_shape == (None, 28, 28, 1)          
    
    elif (Architecture == 5): # Model is CNNVAE  
    
        Encoder_Model = tf.keras.Sequential()
        
        Encoder_Model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        Encoder_Model.add(layers.BatchNormalization())
        Encoder_Model.add(layers.LeakyReLU())
        
        Encoder_Model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        Encoder_Model.add(layers.BatchNormalization())
        Encoder_Model.add(layers.LeakyReLU())    
    
        Encoder_Model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same'))
        Encoder_Model.add(layers.BatchNormalization())
        Encoder_Model.add(layers.LeakyReLU())
    
        Encoder_Model.add(layers.Flatten())
        Encoder_Model.add(layers.Dense(LatentVar_Dimension+LatentVar_Dimension))   
        
        Decoder_Model = tf.keras.Sequential()
        
        Decoder_Model.add(layers.Dense(7*7*128, input_shape=(LatentVar_Dimension,)))
        Decoder_Model.add(layers.BatchNormalization())
        Decoder_Model.add(layers.LeakyReLU())
        
        Decoder_Model.add(layers.Reshape((7, 7, 128)))
        assert Decoder_Model.output_shape == (None, 7, 7, 128)   
        
        Decoder_Model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', ))
        assert Decoder_Model.output_shape == (None, 14, 14, 64)
        Decoder_Model.add(layers.BatchNormalization())
        Decoder_Model.add(layers.LeakyReLU())
        
        Decoder_Model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', ))
        assert Decoder_Model.output_shape == (None, 14, 14, 32)
        Decoder_Model.add(layers.BatchNormalization())
        Decoder_Model.add(layers.LeakyReLU())   
        
        Decoder_Model.add(layers.Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same'))
        assert Decoder_Model.output_shape == (None, 28, 28, 16)
        Decoder_Model.add(layers.BatchNormalization())
        Decoder_Model.add(layers.LeakyReLU()) 
        
        Decoder_Model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same'))   
        
        return Decoder_Model, Encoder_Model
    
    
def Sampling_Decoder(Model, eps=None): 
    
    Decoder_Sample = decode(eps, apply_sigmoid=True)
       
    return Decoder_Sample

def encode(Model, Data): 
     
    mean, logvar = tf.split(Model(Data), num_or_size_splits=2, axis=1) 
     
    return mean, logvar

def Sampling_z( mean, logvar):  
    
    eps = tf.random.normal(shape=mean.shape) 
    
    z = eps * tf.exp(logvar * .5) + mean
     
    return z

def decode( Model, z, apply_sigmoid=False):   
    
    Decoder_Output = Model(z)
      
    if apply_sigmoid:        
      probs = tf.sigmoid(Decoder_Output)      
      return probs  
    
    return Decoder_Output

def Log_Normal_pdf(sample, mean, logvar, raxis=1):
    
    log2pi = tf.math.log(2. * np.pi)
    
    Log_Normal = tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)
    
    return Log_Normal


def VAE_Loss(Encoder_Model, Decoder_Model, Data):
    
    mean, logvar = encode(Encoder_Model, Data)
    
    z = Sampling_z(mean, logvar)
    
    Decoder_Output = decode(Decoder_Model, z)
    
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=Decoder_Output, labels=Data)
    
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    
    logpz = Log_Normal_pdf(z, 0., 0.)
    
    logqz_x = Log_Normal_pdf(z, mean, logvar)
    
    vae_Loss =-tf.reduce_mean(logpx_z + logpz - logqz_x)
    
    return vae_Loss 

def VAE_Training_Step( Encoder_Model, Decoder_Model, VAE_Optimizer, Batch_Data):

    with tf.GradientTape() as tape:
        
        Total_Loss = VAE_Loss(Batch_Data)
      
    VAE_Gradients = tape.gradient(Total_Loss, (Encoder_Model.trainable_variables,Decoder_Model.trainable_variables))
    
    VAE_Optimizer.apply_gradients(zip(VAE_Gradients[0], Encoder_Model.trainable_variables))
    
    VAE_Optimizer.apply_gradients(zip(VAE_Gradients[1], Decoder_Model.trainable_variables,))
    
    return Total_Loss, Encoder_Model, Decoder_Model