#!/usr/bin/env python3
""" Module for implementing sparse autoencoder"""
import tensorflow.keras as keras

def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder
    """
    # Create inputs
    inputs = keras.Input(shape=(input_dims,))
    
    # Build encoder
    encoded = inputs
    for layer_size in hidden_layers:
        encoded = keras.layers.Dense(layer_size, activation='relu')(encoded)
    
    # Create bottleneck
    bottleneck = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoded)
    
    # Create encoder
    encoder = keras.Model(inputs=inputs, outputs=bottleneck)
    
    # Create decoder
    decoder_input = keras.Input(shape=(latent_dims,))
    
    # Build decoder
    decoded = decoder_input
    for layer_size in reversed(hidden_layers):
        decoded = keras.layers.Dense(layer_size, activation='relu')(decoded)
    
    # Output layer
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)
    
    # Create decoder
    decoder = keras.Model(inputs=decoder_input, outputs=decoded)
    
    # Create autoencoder
    auto_output = decoder(encoder(inputs))
    auto = keras.Model(inputs=inputs, outputs=auto_output)
    
    # Compile model
    auto.compile(optimizer='adam', loss='binary_crossentropy')
    
    return encoder, decoder, auto
