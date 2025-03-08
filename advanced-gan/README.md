Folder Organization
```
- advanced-gan
   - auto: autoencoder package
      -autoencoder.py: module containing autoencoder function
   - train_cifar10.py: base
   - 2train_cifar10.py: second experiment (added L1 regularization with lambda)
   - 3train_cifar10.py: second experiment (mirrors the encoder in reverse)
```


# Sparse Autoencoder

This project implements a sparse autoencoder using TensorFlow/Keras. The sparse autoencoder learns compressed representations of data using L1 regularization to enforce sparsity.

## Description

The autoencoder consists of:
- An encoder network that compresses the input data into a lower-dimensional latent space
- A decoder network that reconstructs the original input from the latent representation
- L1 regularization in the latent space to enforce sparsity

The model is demonstrated on the CIFAR-10 dataset, which contains 60,000 32×32 color images across 10 classes.

## Files

- `autoencoder.py`: Contains the implementation of the sparse autoencoder function
- `train_cifar10.py`: Script for training the autoencoder on CIFAR-10 and visualizing results

## Requirements

- Python 3.9
- TensorFlow 2.15
- NumPy 1.25.2
- Matplotlib (for visualization)

## Usage

1. Ensure you have the required dependencies installed
2. Run the training script:
   ```
   python3 train_cifar10.py
   ```

## Function Documentation

The main function provided is:

```
def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder
    
    Parameters:
    input_dims: integer containing the dimensions of the model input
    hidden_layers: list containing the number of nodes for each hidden
                  layer in the encoder, respectively
                  (the hidden layers should be reversed for the decoder)
    latent_dims: integer containing the dimensions of the latent space
                representation
    lambtha: regularization parameter used for L1 regularization on the
            encoded output
    
    Returns:
    encoder: the encoder model
    decoder: the decoder model
    auto: the sparse autoencoder model
    """
```

## Architecture Details

- **Encoder**: Input → Hidden Layers (with ReLU) → Latent Space (with L1 regularization)
- **Decoder**: Latent Space → Reversed Hidden Layers (with ReLU) → Output (with Sigmoid)
- **Optimization**: Adam optimizer with binary cross-entropy loss

## Visualization

The training script generates visualizations including:
- Training and validation loss curves
- Original vs. reconstructed images comparison
- Analysis of latent space activations
