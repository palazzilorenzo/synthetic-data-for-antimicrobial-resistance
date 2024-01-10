#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_vae.py: 
    this module allows to generate new synthetic data using a Variational AutoEncoder.
"""

__author__ = ['Lorenzo Palazzi']
__email__ = ['lorenzo.palazzi@studio.unibo.it']

import os
import sys
sys.path.append(os.getcwd())

import numpy as np

from docs.utils import Get_vae_testing_data
from docs.utils import Load_vae_model
from docs.utils import Predict_vae


def Argv_error():
    message = '''\nYou have to specify the model name.\n 
    Please follow the instructions and provide one or more model names.\n
    Ex.: '--python prediction_vae.py vae_n'
    '''
    return print(message)

def main():
    work_dir = os.getcwd()  # Working directory
    path_to_weights = os.path.join(work_dir,'VAE/weights/')
    
    # get training data
    test, col = Get_vae_testing_data()
    # preparing data to test the VAE
    test = test.to_numpy()
    test = np.expand_dims(test, -1).astype("float32")
    
    # check if latent dimension parameter is provided
    if len(sys.argv)<2:
        Argv_error()
        sys.exit()
        
    # if one argument is provided
    elif len(sys.argv)==2:
        model_name = sys.argv[1]
        vae = Load_vae_model(model_name, path_to_weights)
        Predict_vae(vae, model_name, test, path_to_weights, col)

    else:       
        model_name = sys.argv[1:]
        for name in model_name:
            Predict_vae(name, test, path_to_weights, col)

    
if __name__ == '__main__':

    main()