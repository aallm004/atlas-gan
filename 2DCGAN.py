#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import wandb
import os

batch_size = 128
latent_dim = 100
epochs = 50

# Load and preprocess the MNIST dataset
(train_images, _), (_, _) = mnist.load_data()

# Reshape and normalize the images
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

# create TensowFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images)
train_dataset = train_dataset.shuffle(buffer_size=60000)
train_dataset = train_dataset.batch(batch_size)

def build_generator():
    model = models.Sequential()

    #Upsampling process
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Reshape into 7x7x256 feature maps
    model.add(layers.Reshape((7, 7, 256)))

    # First upsampling block
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Second upsampling block
    model.add(layers.Conv2DTranspose (128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    # Third upsampling block
    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Output layer
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))

    return model

def build_discriminator():
    model = models.Sequential()

    # First layer
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3)) # preventing overfitting

    # Second layer
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Flatten and dense layer for classification
    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # Single output node for binary classification

    return model


# optimizers
generator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
discriminator_optimizer = optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

# loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator = build_generator()
discriminator = build_discriminator()

import time
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


def train_step(real_images, batch_size=batch_size, latent_dim=latent_dim):
    
    # Random noise for generator input
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # Generate fake images
        generated_images = generator(noise, training=True)

        # Discriminator outputs for real and fake images
        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Losses calculation
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        # Gradient calculation
        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        # Application of gradients
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        return gen_loss, disc_loss

def discriminator_loss(real_output, fake_output):
    # Loss for real images should be close to 1
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)

    # Loss for fake images should be close to 0
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    # Total loss is sum of both
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):

    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_images(model, epoch, test_input):

    # generate images
    predictions = model(test_input, training=False)

    # Create a figure to contain plot
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')
        plt.axis('off')

    # Save figure
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close()

def train(dataset, epochs):

    batch_size = 128
    latent_dim = 100
    epochs = 50

    # Generating sample images
    seed = tf.random.normal([16, latent_dim])

    for epoch in range(epochs):
        start = time.time()

        # Initialize metrics for this epoch
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        num_batches = 0

        for image_batch in dataset:
            # Train
            gen_loss, disc_loss = train_step(image_batch)

            # track losses
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss
            num_batches += 1

        # Average loss for this epoch calculation
        epoch_gen_loss /= num_batches
        epoch_disc_loss /= num_batches
        
        # Log to wanb
        wandb.log({
            'epoch': epoch,
            'generator_loss': epoch_gen_loss,
            'discriminator_loss': epoch_disc_loss,
            'time_per_epoch': time.time() - start
        })

        # generate and save images
        if (epoch + 1) % 10 == 0:
            generate_and_save_images(generator, epoch + 1, seed)

            #log images to wandb
            images = generator(seed, training=False)
            images = images * 0.5 + 0.5 # scale from [-1, 1] to [0, 1]
            wandb.log({
                "generated_images": [wandb.Image(img) for img in images]
             })

        #print progress
        print(f'Epoch {epoch+1}, Gen Loss: {epoch_gen_loss:.4f}, '
              f'Disc Loss: {epoch_disc_loss:.4f}, '
              f'Time: {time.time()-start:.2f} sec')

        if (epoch + 1) % 50 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

    generate_and_save_images(generator, epochs, seed)

wandb.init(project="dcgan-mnist", config={
    "learning_rate": 0.0002,
    "batch_size": batch_size,
    "epochs": epochs,
    "latent_dim": latent_dim
})

train(train_dataset, epochs)
