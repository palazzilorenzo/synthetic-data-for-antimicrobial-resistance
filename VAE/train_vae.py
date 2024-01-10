#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_vae.py: 
    this module allows to build and train one or more Variational AutoEncoders.
"""

__author__ = ['Lorenzo Palazzi']
__email__ = ['lorenzo.palazzi@studio.unibo.it']

import os
import sys
sys.path.append(os.getcwd())

import numpy as np

from docs.utils import Get_vae_training_data
from docs.utils import Train_vae

def Argv_error():
    message = '''\nYou have to specify the latent dimensions.\n 
    Please follow the instructions and provide one or more arguments.\n
    Ex.: '--python train_vae.py latent_dim'
    '''
    return print(message)

def main():
    
    # get training data
    train = Get_vae_training_data()
    
    # preparing data to train the VAE
    train = train.to_numpy()
    train = np.expand_dims(train, -1).astype("float32")
    
    # check if latent dimension parameter is provided
    if len(sys.argv)<2:
        Argv_error()
        sys.exit()
        
    # if one argument is provided
    elif len(sys.argv)==2:
        model_name = sys.argv[1]
        Train_vae(model_name, train)
    else:       
        model_name = sys.argv[1:]
        for name in model_name:
            Train_vae(name, train)

    
if __name__ == '__main__':

    main()