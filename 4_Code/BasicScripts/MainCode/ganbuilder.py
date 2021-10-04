# -*- coding: utf-8 -*-
"""

@author: ninad k gaikwad

Module - ganbuilder

"""
# Import Modules
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np



def GANModel_Create(Architecture, Noise_Dimension):
       
    if (Architecture == 1): # Generator_Discriminator_Generator_Model   is DenseGAN
    
        Generator_Model   = tf.keras.Sequential()    
       
        # Deep + Shallow
        Generator_Model.add(layers.Dense(100, input_shape=(Noise_Dimension,)))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())      
        
        # Deep + Shallow
        Generator_Model.add(layers.Dense(300, input_shape=(Noise_Dimension,)))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())  
        
        # Deep + Shallow
        Generator_Model.add(layers.Dense(600))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())       
    
        Generator_Model.add(layers.Dense(784))  # activation = 'tanh in shallow    
    
        Generator_Model.add(layers.Reshape((28, 28, 1)))
        assert Generator_Model.output_shape == (None, 28, 28, 1)   
        
        
        Discriminator_Model  = tf.keras.Sequential()
        
        # Deep + Shallow
        Discriminator_Model.add(layers.Reshape((784,), input_shape=[28, 28, 1]))
        
        # Deep + Shallow
        Discriminator_Model.add(layers.Dense(600, input_shape=(Noise_Dimension,)))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())  
        
        # Deep + Shallow 
        Discriminator_Model.add(layers.Dense(300))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())    
          
        
        # Deep + Shallow
        Discriminator_Model.add(layers.Dense(100)) # activation = 'sigmoid' in shallow
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())     
    
        Discriminator_Model.add(layers.Dense(1))           
    
    elif (Architecture == 2): # Generator_Discriminator_Generator_Model   is DenseGAN_Deep
    
        Generator_Model   = tf.keras.Sequential()    
       
        # Deep + Shallow
        Generator_Model.add(layers.Dense(100, input_shape=(Noise_Dimension,)))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())   
        
        # Deep 
        Generator_Model.add(layers.Dense(200))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())       
        
        # Deep + Shallow
        Generator_Model.add(layers.Dense(300, input_shape=(Noise_Dimension,)))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())
        
        # Deep 
        Generator_Model.add(layers.Dense(400))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())    
        
        # Deep 
        Generator_Model.add(layers.Dense(500))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())   
        
        # Deep + Shallow
        Generator_Model.add(layers.Dense(600))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())       
    
        Generator_Model.add(layers.Dense(784))  # activation = 'tanh in shallow    
    
        Generator_Model.add(layers.Reshape((28, 28, 1)))
        assert Generator_Model.output_shape == (None, 28, 28, 1)   
        
        
        Discriminator_Model  = tf.keras.Sequential()
        
        # Deep + Shallow
        Discriminator_Model.add(layers.Reshape((784,), input_shape=[28, 28, 1]))
        
        # Deep + Shallow
        Discriminator_Model.add(layers.Dense(600, input_shape=(Noise_Dimension,)))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())   
        
        # Deep 
        Discriminator_Model.add(layers.Dense(500))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())       
        
        # Deep 
        Discriminator_Model.add(layers.Dense(400, input_shape=(Noise_Dimension,)))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())
        
        # Deep + Shallow 
        Discriminator_Model.add(layers.Dense(300))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())    
        
        # Deep 
        Discriminator_Model.add(layers.Dense(200))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())   
        
        # Deep + Shallow
        Discriminator_Model.add(layers.Dense(100)) # activation = 'sigmoid' in shallow
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())     
    
        Discriminator_Model.add(layers.Dense(1))                  
    
    elif (Architecture == 3): # Generator_Discriminator_Generator_Model   is CNNGAN
    
        Generator_Model  = tf.keras.Sequential()
        
        Generator_Model.add(layers.Dense(7*7*128, input_shape=(Noise_Dimension,)))
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())
    
        Generator_Model.add(layers.Reshape((7, 7, 128)))
        assert Generator_Model.output_shape == (None, 7, 7, 128)   
    
        Generator_Model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', ))
        assert Generator_Model.output_shape == (None, 14, 14, 64)
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())
        
        Generator_Model.add(layers.Conv2DTranspose(32, (5, 5), strides=(1, 1), padding='same', ))
        assert Generator_Model.output_shape == (None, 14, 14, 32)
        Generator_Model.add(layers.BatchNormalization())
        Generator_Model.add(layers.LeakyReLU())   
    
        Generator_Model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', activation='tanh'))
        assert Generator_Model.output_shape == (None, 28, 28, 1)
        
        Discriminator_Model = tf.keras.Sequential()
        
        Discriminator_Model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())   
    
    
        
        Discriminator_Model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())    
    
        Discriminator_Model.add(layers.Conv2D(128, (5, 5), strides=(1, 1), padding='same'))
        Discriminator_Model.add(layers.BatchNormalization())
        Discriminator_Model.add(layers.LeakyReLU())       
        
    
        Discriminator_Model.add(layers.Flatten())
        Discriminator_Model.add(layers.Dense(1))
        
    
    return Generator_Model  , Discriminator_Model  


def Generator_Loss(Generator_Output):
    
    Cross_Entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    Gen_Loss = Cross_Entropy(tf.ones_like(Generator_Output), Generator_Output)    
    
    return Gen_Loss


def Discriminator_Loss(Generator_Output, Real_Image):
    
    Cross_Entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    RealData_Loss = Cross_Entropy(tf.ones_like(Real_Image,), Real_Image,)
    GeneratorData_Loss = Cross_Entropy(tf.zeros_like(Generator_Output), Generator_Output)
    
    Discriminator_Loss = RealData_Loss + GeneratorData_Loss
    
    return Discriminator_Loss   

  
def GAN_Training_Step(Generator_Model, Discriminator_Model, Generator_Optimizer, Discriminator_Optimizer, Batch_Size, Noise_Dimension, Real_Data):
    
    z = tf.random.normal([Batch_Size, Noise_Dimension])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    
        Generated_Images = Generator_Model(z, training=True)
        
        Real_Image_Output = Discriminator_Model(Real_Data, training=True)
        Generator_Image_Output = Discriminator_Model(Generated_Images, training=True)
        
        gen_loss = Generator_Loss(Generator_Image_Output)
        dis_loss = Discriminator_Loss(Real_Image_Output, Generator_Image_Output)
    
    Generator_Gradients = gen_tape.gradient(gen_loss, Generator_Model.trainable_variables)
    Discriminator_Gradients = disc_tape.gradient(dis_loss, Discriminator_Model.trainable_variables)
    
    Generator_Optimizer.apply_gradients(zip(Generator_Gradients, Generator_Model.trainable_variables))
    Discriminator_Optimizer.apply_gradients(zip(Discriminator_Gradients, Discriminator_Model.trainable_variables))
    
    return gen_loss, dis_loss, Generator_Model, Discriminator_Model   
