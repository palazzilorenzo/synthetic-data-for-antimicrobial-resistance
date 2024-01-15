"""
prediction_vae.py: 
    this module allows to generate new synthetic data using both CVAE and VAE.
"""

__author__ = ['Lorenzo Palazzi']
__email__ = ['lorenzo.palazzi@studio.unibo.it']

import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd

from docs.utils import Get_testing_data, Select_susc_spectra, Select_res_spectra
from docs.utils import Load_model
from docs.utils import Predict_data


def Argv_error():
    message = '''\nYou have to specify the model.\n 
    Please follow the instructions and provide one or more model/s.\n
    Ex.: '--python generate_data.py model_dim'
    '''
    return print(message)

def main():
    work_dir = os.getcwd()  # Working directory
    # check if latent dimension parameter is provided
    if len(sys.argv)<2:
        Argv_error()
        sys.exit()
    # if one argument is provided
    elif len(sys.argv)==2:
        model_dim = sys.argv[1]
        model_name = model_dim.split('_')[0]
        if model_name=='vae':
            path_to_weights = os.path.join(work_dir,f'{model_name}/weights/')
            # get testing data
            test, col = Get_testing_data(model_name)
            # preparing data to test the VAE
            test = test.to_numpy()
            test = np.expand_dims(test, -1).astype("float32")
            vae = Load_model(model_dim, path_to_weights)
            Predict_data(vae, model_dim, test, col)
        elif model_name=='cvae':
            path_to_weights = os.path.join(work_dir,f'{model_name}/weights/')
            cvae = Load_model(model_dim, path_to_weights)
            # get training data and divide them into susceptible and resistant spectra
            test_x, test_labels, test, col = Get_testing_data(model_name)
            Predict_data(cvae, model_dim, test_x, col, labels=test_labels)
            print('\n Generate only susceptible spectra: \n')
            test_x_susc, test_labels_susc = Select_susc_spectra(test)
            Predict_data(cvae, model_dim, test_x_susc, col, labels=test_labels_susc, susc=1)
            print('\n Generate only resistant spectra: \n')
            test_x_res, test_labels_res = Select_res_spectra(test)
            Predict_data(cvae, model_dim, test_x_res, col, labels=test_labels_res, susc=0)

    else:       
        model_dim = sys.argv[1:]
        for model in model_dim:
            model_name = model.split('_')[0]
            if model_name=='vae':
                path_to_weights = os.path.join(work_dir,f'{model_name}/weights/')
                # get testing data
                test, col = Get_testing_data(model_name)
                # preparing data to test the VAE
                test = test.to_numpy()
                test = np.expand_dims(test, -1).astype("float32")
                vae = Load_model(model, path_to_weights)
                Predict_data(vae, model, test, col)
            elif model_name=='cvae':
                path_to_weights = os.path.join(work_dir,f'{model_name}/weights/')
                cvae = Load_model(model, path_to_weights)
                # get training data and divide them into susceptible and resistant spectra
                test_x, test_labels, test, col = Get_testing_data(model_name)
                Predict_data(cvae, model, test_x, col, labels=test_labels)
                print('\n Generate only susceptible spectra: \n')
                test_x_susc, test_labels_susc = Select_susc_spectra(test)
                Predict_data(cvae, model, test_x_susc, col, labels=test_labels_susc, susc=1)
                print('\n Generate only resistant spectra: \n')
                test_x_res, test_labels_res = Select_res_spectra(test)
                Predict_data(cvae, model, test_x_res, col, labels=test_labels_res, susc=0)
            
    
if __name__ == '__main__':

    main()
