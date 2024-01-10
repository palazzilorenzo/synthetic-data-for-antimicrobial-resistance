#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_cvae.py: 
    this module allows to build and train one or more Conditional Variational AutoEncoders.
"""

__author__ = ['Lorenzo Palazzi']
__email__ = ['lorenzo.palazzi@studio.unibo.it']

import os
import sys
sys.path.append(os.getcwd())

from docs.utils import Get_cvae_training_data
from docs.utils import Train_cvae

def Argv_error():
    message = '''\nYou have to specify the latent dimensions.\n 
    Please follow the instructions and provide one or more arguments.\n
    Ex.: '--python train_vae.py latent_dim'
    '''
    return print(message)

def main():
    
    # get training data
    train_x, train_labels = Get_cvae_training_data()
    
    # check if latent dimension parameter is provided
    if len(sys.argv)<2:
        Argv_error()
        sys.exit()
        
    # if one argument is provided
    elif len(sys.argv)==2:
        model_name = sys.argv[1]
        Train_cvae(model_name, train_x, train_labels)
    else:       
        model_name = sys.argv[1:]
        for name in model_name:
            Train_cvae(name, train_x, train_labels)

    
if __name__ == '__main__':

    main()