#!/usr/bin/env python3
"""
Training script for sparse autoencoder using CIFAR-10 with Weights & Biases
"""
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import wandb
import time
import os
from auto.autoencoder import autoencoder

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

batch_size = 128
latent_dim = 64
epochs = 20

# Load CIFAR-10 dataset
(x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

# Reshape and normalize images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten images
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Create model
input_dims = 32 * 32 * 3
hidden_layers = [512, 256, 128]
lambtha = 1e-4

wandb.init(project="sparse-autoencoder-cifar10", config={
    "learning_rate": 0.001,
    "batch_size": batch_size,
    "epochs": epochs,
    "latent_dim": latent_dim,
    "hidden_layers": hidden_layers,
    "lambtha": lambtha
})

# Create models
encoder, decoder, auto = autoencoder(input_dims, hidden_layers, latent_dim, lambtha)

def generate_and_save_images(encoder, decoder, epoch, test_images):
    """function to generate and save reconstructed images"""    

    test_images_flat = test_images.reshape(test_images.shape[0], -1)
    encoded = encoder.predict(test_images_flat)
    reconstructed = decoder.predict(encoded)
    reconstructed = reconstructed.reshape(test_images.shape)
    
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(reconstructed[i])
        plt.axis('off')
    
    plt.savefig('reconstruction_at_epoch_{:04d}.png'.format(epoch))
    plt.close()
    
    wandb.log({
        "reconstructed_images": [wandb.Image(img) for img in reconstructed[:16]]
    })

def train(x_train, x_test, batch_size, epochs):
    sample_images = x_test[:16]
    
    # Create a dataset from training data
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train_flat)
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)
    
    for epoch in range(epochs):
        start = time.time()

        epoch_loss = 0
        num_batches = 0
        
        for batch in train_dataset:
            history = auto.fit(batch, batch, epochs=1, verbose=0)
            
            epoch_loss += history.history['loss'][0]
            num_batches += 1
        
        epoch_loss /= num_batches
        
        val_loss = auto.evaluate(x_test_flat, x_test_flat, verbose=0)
        
        wandb.log({
            'epoch': epoch,
            'loss': epoch_loss,
            'val_loss': val_loss,
            'time_per_epoch': time.time() - start
        })
        
        if (epoch + 1) % 5 == 0:
            generate_and_save_images(encoder, decoder, epoch + 1, sample_images)
        
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Time: {time.time()-start:.2f} sec')
    
    generate_and_save_images(encoder, decoder, epochs, sample_images)

train(x_train, x_test, batch_size, epochs)
