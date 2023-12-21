#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 12:38:39 2023

@author: Lorenzo Palazzi, University of Bologna - Applied Physics, lorenzo.palazzi@studio.unibo.it
"""
from docs.sampling import Sampling
from docs.class_cvae import CVAE

from keras import Model
from keras.layers import Input, Dense, Conv1D, Flatten, Reshape, Conv1DTranspose, Concatenate
import keras.backend as K
from keras.optimizers import Adam

def Cvae(latent_dim, input_shape=6000, n_labels=2):
    """" """
    ### ENCODER ###
    # defining input layers
    X = Input(shape=(input_shape, 1))
    label = Input(shape=(n_labels, 1))
    # merge data representation and label
    encoder_inputs = Concatenate(axis=1)([X, label])
    x = Conv1D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = Conv1D(64, 3, activation="relu", strides=1, padding="same")(x)
    shape_x = K.int_shape(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model([X, label], [z_mean, z_log_var, z], name=f"encoder_{latent_dim}")
    ### DECODER ###
    latent_inputs = Input(shape=(latent_dim,))
    label = Reshape((2,))(label)
    x = Concatenate(axis=1)([latent_inputs, label])
    x = Dense(1 * shape_x[1] * shape_x[2], activation="relu")(x)
    x = Reshape((shape_x[1], shape_x[2]))(x)
    x = Conv1DTranspose(64, 3, activation="relu", strides=1, padding="same")(x)
    x = Conv1DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv1DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = Model([latent_inputs, label], decoder_outputs, name=f"decoder_{latent_dim}")
    # build and compile model
    model = CVAE(encoder, decoder)
    model.compile(optimizer=Adam())
    return model