# -*- coding: utf-8 -*-
"""

@author: ninad k gaikwad

Module - trainingplotter

"""
# Import Modules
import matplotlib.pyplot as plt
from vaebuilder import *


def Plot_Training_GAN(GAN_Model, Epoch, Test_Image, Gen_Loss_Store, Disc_Loss_Store, Epoch_Store):

    # Getting Output from GAN Generator
    Output_Image = GAN_Model(Test_Image, training=False)
    
    # Plotting Generator and Discriminator Loss vs. Epoch
    plt.figure()
      
    plt.plot(Epoch_Store,Gen_Loss_Store, label="Generator Loss")
    plt.plot(Epoch_Store,Disc_Loss_Store, label="Discriminator Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()   
    
    # Plotting New Image from Generator
    plt.figure(figsize=(4,4))
    
    for i in range(Output_Image.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(Output_Image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
      
    plt.show()
    
    return
  
  
def Plot_Training_VAE(VAE_Encoder_Model, Epoch, Test_Image, Loss_Store, Epoch_Store):
  
    Mean, Logvar = VAE_Encoder_Model(Test_Image)
    
    z = vaebuilder.Sampling_z(Mean, Logvar)
    
    Output_Image = vaebuilder.Sampling_Decoder(z)
    
    # Plotting Elbo Loss vs. Epoch
    plt.figure()    
    plt.plot(Epoch_Store, Loss_Store, label="ELBO LOSS")  
    plt.ylabel('ElBO Loss')
    plt.xlabel('Epoch')
    plt.show()  
    
    plt.figure(figsize=(4, 4))
    
    # Plotting New Image from Decoder
    for i in range(Output_Image.shape[0]):      
        plt.subplot(4, 4, i + 1)
        plt.imshow(Output_Image[i, :, :, 0], cmap='gray')
        plt.axis('off')
      
    plt.show()   
    
    return

    
