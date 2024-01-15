#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_model.py: 
    this module allows to build and train one or more Conditional Variational/Variational AutoEncoders.
"""

__author__ = ['Lorenzo Palazzi']
__email__ = ['lorenzo.palazzi@studio.unibo.it']

import os
import sys
sys.path.append(os.getcwd())

import numpy as np

from docs.utils import Get_training_data
from docs.utils import Train_model

def Argv_error():
    message = '''\nYou have to specify which model to train.\n 
    Please follow the instructions and provide one or more arguments.\n
    Ex.: '--python train.py cvae_64'
    '''
    return print(message)

def main():
    # check if latent dimension parameter is provided
    if len(sys.argv)<2:
        Argv_error()
        sys.exit()
    # if one argument is provided
    elif len(sys.argv)==2:
        model_dim = sys.argv[1]
        model_name = model_dim.split('_')[0]
        if model_name=='vae':
            # get training data
            train = Get_training_data(model_name)
            # preparing data to train the VAE
            train = train.to_numpy()
            train = np.expand_dims(train, -1).astype("float32")
            # start training
            Train_model(model_dim, train)
        elif model_name=='cvae':
            # get training data
            train, labels = Get_training_data(model_name)
            # start training
            Train_model(model_dim, train, labels)
            
    else:       
        model_dim = sys.argv[1:]
        for model in model_dim:
            model_name = model.split('_')[0]
            if model_name=='vae':
                # get training data
                train = Get_training_data(model_name)
                # preparing data to train the VAE
                train = train.to_numpy()
                train = np.expand_dims(train, -1).astype("float32")
                Train_model(model, train)
            elif model_name=='cvae':
                # get training data
                train, labels = Get_training_data(model_name)
                # start training
                Train_model(model, train, labels)
    
if __name__ == '__main__':

    main()