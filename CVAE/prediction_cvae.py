#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prediction_cvae.py: 
    this module allows to generate new synthetic data using a Conditional Variational AutoEncoder.
"""

__author__ = ['Lorenzo Palazzi']
__email__ = ['lorenzo.palazzi@studio.unibo.it']

import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
from keras.utils import to_categorical

from docs.utils import Get_cvae_testing_data
from docs.utils import Load_cvae_model
from docs.utils import Predict_cvae, Predict_cvae_susc, Predict_cvae_res

def Argv_error():
    message = '''\nYou have to specify the model name.\n 
    Please follow the instructions and provide one or more model names.\n
    Ex.: '--python prediction_cvae.py cvae_n'
    '''
    return print(message)

def main():
    work_dir = os.getcwd()  # Working directory
    path_to_weights = os.path.join(work_dir,'CVAE/weights/')
    
    # get training data and divide them into susceptible and resistant spectra
    test_x, test_labels, test, col = Get_cvae_testing_data()
    
    test_susc = test.loc[test['Ampicillin'] == 1] # select only susceptible spectra
    test_x_susc = test_susc.drop(['Ampicillin'], axis = 1) # intensity values
    test_labels_susc = pd.DataFrame(test_susc['Ampicillin']) # labels
    test_labels_susc = to_categorical(test_labels_susc, num_classes=2)
    
    test_res = test.loc[test['Ampicillin'] == 0] # select only resistant spectra
    test_x_res = test_res.drop(['Ampicillin'], axis = 1) # intensity values
    test_labels_res = pd.DataFrame(test_res['Ampicillin']) # labels
    test_labels_res = to_categorical(test_labels_res, num_classes=2)
    
    # check if latent dimension parameter is provided
    if len(sys.argv)<2:
        Argv_error()
        sys.exit()
        
    # if one argument is provided
    elif len(sys.argv)==2:
        model_name = sys.argv[1]
        cvae = Load_cvae_model(model_name, path_to_weights)
        Predict_cvae(cvae, model_name, test_x, test_labels, col)
        Predict_cvae_susc(cvae, model_name, test_x_susc, test_labels_susc, col)
        Predict_cvae_res(cvae, model_name, test_x_res, test_labels_res, col)
    else:       
        model_name = sys.argv[1:]
        cvae_list = []
        for name in model_name:
            cvae = Load_cvae_model(name, path_to_weights)
            cvae_list.append(cvae) 
        for cvae, name in  zip(cvae_list, model_name):    
            Predict_cvae(cvae, name, test_x, test_labels, col)
            Predict_cvae_susc(cvae, name, test_x_susc, test_labels_susc, col)
            Predict_cvae_res(cvae, name, test_x_res, test_labels_res, col)
    
if __name__ == '__main__':

    main()