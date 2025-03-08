import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tensorflow_datasets as tfds

# Hyperparameters
batch_size = 64
latent_dim = 128
epochs = 50
learning_rate = 0.0002
beta_1 = 0.5

# Image dimensions
img_height = 64
img_width = 64
channels = 3

# Home for results
results_dir = './flowers_gan_results'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/images', exist_ok=True)
checkpoint_dir = f'{results_dir}/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

def load_flowers_dataset(batch_size, img_height, img_width):
    """ Load and prepare places"""

    print("Loading TensorFlow Flowers dataset...")

    # Load the tf_flowers dataset
    flowers_dataset, info = tfds.load('tf_flowers', split='train', with_info=True)
    num_examples = info.splits['train'].num_examples
    print(f"Dataset has {num_examples} examples")


    def preprocess_image(data):
        # Extract image from the data dictionary
        image = data['image']

        # Resize image to target destination
        image = tf.image.resize(image, [img_height, img_width])

        # Normalize to [-1, 1] range
        image = (tf.cast(image, tf.float32) - 127.5) / 127.5

        return image

    train_dataset = flowers_dataset(preprocess_image).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, num_examples

def build_generator():
    """Build generator model for Places dataset"""
    model = models.Sequential()

    # Foundation for 8x8 maps
    model.add(layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(latent_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Reshape to 8x8x512
    model.add(layers.Reshape((8, 8, 512)))

    # Upsampling block 1 8x8 to 16x16
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsampling block 2 16x16 to 32x32
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Upsampling block 3 32x32 to 64x64
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    # Output layer
    model.add(layers.Conv2D(3, (5, 5), padding='same', use_bias=False, activation='tanh'))

    return model

def build_discriminator():
    """Build discriminator model for Places dataset"""
    model=models.Sequential()

    # First conv layer
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[img_height, img_width, channels]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Second conv layer
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Third conv layer
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Fourth conv layer
    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    # Classification
    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
discriminator_optimizer = optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)

# Set up checkpoint
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

def generate_and_save_images(model, epoch, test_input):
    """Generate and save sample images from the generator"""
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(8, 8))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        # Converting the pixel values from [-1, 1] to [0, 1]
        img = (predictions[i, :, :, :] + 1) / 2.0
        plt.imshow(img)
        plt.axis('off')

    plt.savefig(f'{results_dir}/images/image_at_epoch_{epoch:04d}.png')
    plt.close()

@tf.function
def train_step(images):
    """single training step for both gen and disc"""
    # Get actual batch size (last batch might be smaller)
    current_batch_size = tf.shape(images)[0]
    
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        generated_images = generator(noise, training=True)

        # Get discriminator outputs
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        # Calculate losses
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

def train(dataset, epochs):
    """Train the GAN for the speficied number of epochs"""
    # Seed for generating sample images
    seed = tf.random.normal([16, latent_dim])

    # Log file
    log_file = open(f'{results_dir}/training_log.txt', 'w')
    log_file.write("Epoch, Gen Loss, Disc Loss, Time (sec)\n")

    # Display initial generation
    generate_and_save_images(generator, 0, seed)

    # Calculate steps per epoch
    steps_per_epoch = num_examples // batch_size

    for batch_images in dataset:
        start = time.time()

        # Track metrics for this epoch
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        num_batches = 0

        # Train on all batches for this epoch
        for batch_images in dataset:
            # Train step
            gen_loss, disc_loss = train_step(batch_images)

            # Track losses
            epoch_gen_loss += gen_loss
            epoch_disc_loss += disc_loss
            num_batches += 1

            # Print progress every 10 batches
            if num_batches % 10 == 0:
                print(f'Epoch {epoch+1}, Batch {num_batches}/{steps_per_epoch}')

        # Calculate average losses
        epoch_gen_loss /= num_batches
        epoch_disc_loss /= num_batches

        # Calculate time taken
        time_taken = time.time() - start
        
        # Log metrics
        log_file.write(f"{epoch+1}, {float(epoch_gen_loss):.4f}, {float(epoch_disc_loss):.4f}, {time_taken:.2f}\n")
        log_file.flush()
        
        # Generate sample images
        if (epoch + 1) % 5 == 0 or epoch == 0:
            generate_and_save_images(generator, epoch + 1, seed)
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            
        # Print progress
        print(f'Epoch {epoch+1}/{epochs}, Gen Loss: {float(epoch_gen_loss):.4f}, '
              f'Disc Loss: {float(epoch_disc_loss):.4f}, Time: {time_taken:.2f} sec')
    
    # Generate final images
    generate_and_save_images(generator, epochs, seed)
    
    # Save final checkpoint
    checkpoint.save(file_prefix=checkpoint_prefix)
    
    # Close log file
    log_file.close()
    
    print(f"Training completed! Results saved in {results_dir}")

if __name__ == "__main__":
    # Make sure TensorFlow is using GPU if available
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print(f"Using GPU: {physical_devices[0].name}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("No GPU found, using CPU")
    
    # Load the flowers dataset
    flowers_dataset, num_examples = load_flowers_dataset(batch_size, img_height, img_width)
    
    # Train the GAN
    train(flowers_dataset, num_examples, epochs)
