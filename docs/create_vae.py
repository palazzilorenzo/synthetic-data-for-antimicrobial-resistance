"""
Created on Thu Dec 14 14:00:07 2023

@author: Lorenzo Palazzi, University of Bologna - Applied Physics, lorenzo.palazzi@studio.unibo.it
"""
from docs.sampling import Sampling
from docs.class_vae import VAE

from keras import Model
from keras.layers import Input, Dense, Conv1D, Flatten, Reshape, Conv1DTranspose
import keras.backend as K
from keras.optimizers import Adam

def Vae(latent_dim, input_shape=6000):
    """" """
    # defining encoder
    encoder_inputs = Input(shape=(input_shape, 1))
    x = Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = Conv1D(64, 3, activation="relu", strides=2, padding="same")(x)
    shape_x = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name=f"encoder_{latent_dim}")
    # defining decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(1 * shape_x[1] * shape_x[2], activation="relu")(latent_inputs)
    x = Reshape((shape_x[1], shape_x[2]))(x)
    x = Conv1DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name=f"decoder_{latent_dim}")
    # build and compile model
    model = VAE(encoder, decoder)
    model.compile(optimizer=Adam())
    return model

