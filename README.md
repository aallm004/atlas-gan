# Atlas GAN: CIFAR-10 Image Generation

A Deep Convolutional Generative Adversarial Network (DCGAN) implementation for generating CIFAR-10 images, exploring advanced GAN training dynamics and architectural variations.

## Project Overview

This project implements a DCGAN architecture to tackle the challenging task of generating realistic CIFAR-10 images. Unlike simpler datasets like MNIST, CIFAR-10 presents significant complexity with its color images and diverse object classes (airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, trucks).

**Key Achievements:**
- Successfully generated recognizable CIFAR-10-like images with proper color distributions
- Conducted systematic architectural and hyperparameter experiments
- Documented and analyzed mode collapse phenomenon in GANs
- Achieved stable training for ~50 epochs before encountering training dynamics challenges

## Architecture

### DCGAN Implementation
- **Generator**: Transposed convolutional layers with batch normalization and ReLU activation
- **Discriminator**: Convolutional layers with LeakyReLU activation and dropout
- **Latent Dimension**: 100
- **Framework**: TensorFlow/Keras

### Architectural Variations Tested
- **Filter Counts**: Increased from 64-128-256 to 128-256-512 (improved results)
- **Network Depth**: Experimented with additional convolutional layers
- **Kernel Sizes**: Various configurations tested for optimal performance

## Experimental Approach

### Baseline Configuration
```python
# Initial hyperparameters
latent_dim = 100
batch_size = 64
learning_rate = 0.0002  # Both networks
optimizer = Adam
loss_function = binary_crossentropy
```

### Advanced Configuration
```python
# Optimized hyperparameters
batch_size = 32
generator_lr = 0.0001
discriminator_lr = 0.0004
dropout = 0.4
label_smoothing = True
filter_progression = [128, 256, 512]
```

## Results & Key Findings

### Training Dynamics
- **Epochs 1-50**: Stable training with improving image quality
- **Epoch ~70**: Mode collapse occurred in extended training runs
- **Best Results**: Achieved around epoch 50 with architectural improvements

### Mode Collapse Analysis
The project provided valuable insights into mode collapse phenomenon:
- **Trigger**: Discriminator became too effective (learning rate imbalance)
- **Symptoms**: Generator loss increased from ~1.1 to >2.0, discriminator loss dropped below 0.7
- **Visual Impact**: Generated images showed similar patterns and reduced diversity

### Performance Metrics
- Generated images with appropriate CIFAR-10 color distributions
- Recognizable object shapes across different classes
- Improved sharpness and boundary definition through architectural optimization

## Getting Started

### Prerequisites
```bash
pip install tensorflow
pip install numpy
pip install matplotlib
pip install PIL
```

### Usage
```bash
# Clone the repository
git clone https://github.com/aallm004/atlas-gan.git
cd atlas-gan

# Run training
python train_gan.py

# Generate samples
python generate_samples.py
```

## Training Progress

The training process demonstrates the delicate balance required in GAN optimization:

1. **Early Training (Epochs 1-20)**: Networks learn basic features and color patterns
2. **Stable Phase (Epochs 20-50)**: Best quality images generated with good diversity
3. **Instability (Epochs 50+)**: Risk of mode collapse increases with extended training

## Key Learnings

### Technical Insights
- **Generator-Discriminator Balance**: Critical for stable training
- **Extended Training Risks**: Longer doesn't always mean better
- **Architectural Impact**: Filter count increases improved image quality more than depth
- **Hyperparameter Sensitivity**: Learning rate ratios significantly affect training dynamics

### Mode Collapse Prevention Strategies
Future improvements identified through this project:
- Minibatch discrimination for diversity encouragement
- Spectral normalization to prevent discriminator dominance
- Alternative loss functions (Wasserstein, LSGAN)
- Progressive growing techniques

## Project Structure
```
atlas-gan/
├── train_gan.py              # Main training script
├── models/
│   ├── generator.py          # Generator architecture
│   └── discriminator.py      # Discriminator architecture
├── utils/
│   ├── data_loader.py        # CIFAR-10 data preprocessing
│   └── visualization.py     # Training progress visualization
├── samples/                  # Generated image samples
└── checkpoints/             # Model checkpoints
```

## Technical Implementation

### Data Preprocessing
- CIFAR-10 images normalized to [-1, 1] range
- Real-time data augmentation during training
- Batch shuffling and proper train/validation splits

### Loss Functions
- **Generator Loss**: Binary cross-entropy with inverted labels
- **Discriminator Loss**: Binary cross-entropy with real/fake classification
- **Label Smoothing**: Applied to reduce discriminator overconfidence

## Future Enhancements

Based on the mode collapse analysis, planned improvements include:
- **Wasserstein GAN**: More stable training dynamics
- **Progressive GAN**: Gradual resolution increase
- **Self-Attention**: Better spatial relationship modeling
- **Spectral Normalization**: Lipschitz constraint enforcement

---

*This project demonstrates advanced understanding of generative modeling challenges and provides practical insights into GAN training dynamics for complex image datasets.*
