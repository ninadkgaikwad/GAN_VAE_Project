# -*- coding: utf-8 -*-
"""
@author: ninad k gaikwad

VAE - with Fully Connected Layers and no CNNs using Keras+TensorFlow

"""

## Required Module
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import time

## Data-Training Control
train_size = 60000
batch_size = 128
test_size = 1000
epochs = 50
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16

## Getting MINST Data
(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

## Pre-Processing MINST Data
def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

## Buffer and Batch Training Data
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images).shuffle(test_size).batch(batch_size))

## Creating Models

# Encoder Model
def make_encoder_model():
    
    model = tf.keras.Sequential()
    
    model.add(layers.Reshape((784,), input_shape=[28, 28, 1]))    
    
    model.add(layers.Dense(600))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(300))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())    

    model.add(layers.Dense(100))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(50))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())    

    model.add(layers.Dense(latent_dim+latent_dim))
    
    return model   

   
# Decoder Model
def make_decoder_model():
    
    model = tf.keras.Sequential()
    
    model.add(layers.Dense(50, input_shape=[latent_dim]))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(100))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())    

    model.add(layers.Dense(300))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    model.add(layers.Dense(600))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())     
    
    model.add(layers.Dense(784))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())  

    model.add(layers.Reshape((28, 28, 1)))
    assert model.output_shape == (None, 28, 28, 1)     
    
    return model

           
# Genrating Encoder and Decoder Models
encoder=make_encoder_model()
decoder=make_decoder_model()


def sample( eps=None):      
  if eps is None:        
    eps = tf.random.normal(shape=(100, latent_dim))      
  return decode(eps, apply_sigmoid=True)

def encode( x):      
  mean, logvar = tf.split(encoder(x), num_or_size_splits=2, axis=1)    
  return mean, logvar

def reparameterize( mean, logvar):      
  eps = tf.random.normal(shape=mean.shape)    
  return eps * tf.exp(logvar * .5) + mean

def decode( z, apply_sigmoid=False):      
  logits = decoder(z)    
  if apply_sigmoid:        
    probs = tf.sigmoid(logits)      
    return probs  
  return logits

## Testing Untrained Models


# Pick a sample of the test set for generating output images
#assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
    
    test_sample = test_batch[0:num_examples_to_generate, :, :, :]

#generate_and_save_images(model, 0, test_sample)  

## Defining Loss Functions

optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    
  log2pi = tf.math.log(2. * np.pi)
  
  return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi), axis=raxis)


def compute_loss( x):
    
  mean, logvar = encode(x)
  
  z = reparameterize(mean, logvar)
  
  x_logit = decode(z)
  
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  
  logpz = log_normal_pdf(z, 0., 0.)
  
  logqz_x = log_normal_pdf(z, mean, logvar)
  
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

## Defining Training Step
  
#@tf.function
def train_step(x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
      
    loss = compute_loss(x)
    
  gradients = tape.gradient(loss, (encoder.trainable_variables,decoder.trainable_variables))
  
  optimizer.apply_gradients(zip(gradients[0], encoder.trainable_variables))
  
  optimizer.apply_gradients(zip(gradients[1], decoder.trainable_variables,))
  
  return loss
  
# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])

def generate_and_save_images(epoch, test_sample, Loss_Store, Epoch_Store):
    
  mean, logvar = encode(test_sample)
  
  z = reparameterize(mean, logvar)
  
  predictions = sample(z)
  
  plt.figure()
    
  plt.plot(Epoch_Store,Loss_Store, label="ELBO LOSS")
  
  plt.ylabel('ElBO Loss')
  plt.xlabel('Epoch')
  #plt.legend()
  plt.show()  
  
  plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
      
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
    
  plt.show()    
    
  plt.figure(figsize=(4, 4))

  for i in range(test_sample.shape[0]):
      
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_sample[i, :, :, 0], cmap='gray')
    plt.axis('off')       

  # tight_layout minimizes the overlap between 2 sub-plots
  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

## Defining Traing Loop

epoch_num = 0

epoch_store = []

loss_store = []
  
time_epoch_store = [] 

for epoch in range(1, epochs + 1):
    
  epoch_num = epoch_num+1
    
  epoch_store.append(epoch_num)    
    
  start_time = time.time()
  
  for train_x in train_dataset:
      
    loss = train_step(train_x, optimizer)
  
    end_time = time.time()  
    
  loss_store.append(loss.numpy())
  
  end_time = time.time()
    
  time_epoch = end_time - start_time 
    
  time_epoch_store.append(time_epoch)
    
  time_epochs_total = sum(time_epoch_store)     

  print ('Time for epoch {} is {} sec'.format(epoch + 1, time_epoch))
  print ('Total time = {} sec'.format(time_epochs_total))
  print ('ELBO Loss = {}  '.format(loss))     
    
  # loss = tf.keras.metrics.Mean()
  
  # for test_x in test_dataset:
      
  #   loss(compute_loss(model, test_x))
    
  # elbo = -loss.result()
  
  # display.clear_output(wait=False)  
   
  generate_and_save_images(epoch, test_sample, loss_store, epoch_store)
  





