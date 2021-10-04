# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 22:54:36 2020

@author: ninad
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 21:53:54 2020

@author: ninad

GAN - First iteration - Using Keras+TensorFlow

"""

## Required Modules
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

## Data-Training Control
BUFFER_SIZE = 1000
BATCH_SIZE = 8
EPOCHS = 10
noise_dim = 1
num_examples_to_generate = 16
# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

## Getting MINST Data
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

## Pre-Processing MINST Data
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

## Buffer and Batch Training Data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

## Creating Models

# Generator Model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(100, input_shape=(1,)))
    model.add(layers.Dense(300, input_shape=(100,)))
    model.add(layers.Dense(600, input_shape=(300,)))
    model.add(layers.Dense(784, input_shape=(600,)))
    #assert model.output_shape == (None, 28, 28, 1)

    return model

# Discriminator Model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

## Testing Untrained Models

# Testing: Untrained Generator Model
generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')    

# Testing: Untrained Discriminator Model
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

## Defining Loss Function

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Defining Generator Loss Function
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Defining Discriminator Loss Function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

## Defining Optimizers
# The discriminator and the generator optimizers are different since we will train two networks separately.

# Generator Optimizer
generator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Discriminator Optimizer
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

## Save checkpoints

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

## Defining Train - Single Step
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

## Image Functions
  
# Generate and Save Image
  
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
    
  plt.savefig('GAN_image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()
  
  
##Create a GIF
# Display a single image using the epoch number
  
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))



## Defining Training Loop
    
def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    #display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  #display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
  
  
## Training the Model  
train(train_dataset, EPOCHS)

## Display Epoch Image
display_image(EPOCHS)
  

