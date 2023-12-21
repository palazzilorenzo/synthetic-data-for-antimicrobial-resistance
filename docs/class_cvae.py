#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:44:45 2023

@author: Lorenzo Palazzi, University of Bologna - Applied Physics, lorenzo.palazzi@studio.unibo.it
"""

import tensorflow as tf

from keras import Model
from keras.layers import Reshape, Concatenate
from keras.metrics import binary_crossentropy, Mean
from tensorflow.math import reduce_mean, reduce_sum


class CVAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]


    def train_step(self, data):
        with tf.GradientTape() as tape:
            inputs, y_labels = data
            n_x = inputs.shape[1]
            n_labels = y_labels.shape[1]
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder([z, y_labels])
            #################################### NEW LINES
            concatenate_data = Concatenate(axis=1)((inputs, y_labels))
            concatenate_data = Reshape((n_x+n_labels, 1))(concatenate_data)
            ####################################
            reconstruction_loss = reduce_mean(
                reduce_sum(
                    binary_crossentropy(concatenate_data, reconstruction), axis=1
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }