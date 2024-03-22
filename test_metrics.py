#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_metrics.py: 
    this module allows to test the quality of new synthetic data w.r.t. real data,
    using consolidated metrics. 
"""

__author__ = ['Lorenzo Palazzi']
__email__ = ['lorenzo.palazzi@studio.unibo.it']

import os
import sys
sys.path.append(os.getcwd())

from docs.utils import Get_testing_data, Get_synthetic_data, Get_metadata_dict
from docs.utils import Select_susc_spectra, Select_res_spectra
from docs.utils import Generate_quality_report, Generate_diagnostic_report


def Generate_vae_reports(prediction_model_dim, metadata_dict):
    model_name = prediction_model_dim.split('_')[1]
    real_data, _ = Get_testing_data(model_name)
    synthetic_data = Get_synthetic_data(prediction_model_dim)
    ''' 
    To reduce computational time we'll test only the range 4400-5600.
    If you want to test the entire set, comment the following two lines. 
    '''
    real_data = real_data.iloc[:, 800:1201]
    synthetic_data = synthetic_data.iloc[:, 800:1201]
    
    Generate_quality_report(prediction_model_dim, real_data, synthetic_data, metadata_dict)
    Generate_diagnostic_report(prediction_model_dim, real_data, synthetic_data, metadata_dict)
    
def Generate_cvae_reports(prediction_model_dim, metadata_dict):
    model_name = prediction_model_dim.split('_')[1]
    _, _, real_data, _ = Get_testing_data(model_name)
    synthetic_data = Get_synthetic_data(prediction_model_dim)
    
    if len(prediction_model_dim.split('_')) == 3:
        ''' 
        To reduce computational time we'll test only the range 4400-5600.
        If you want to test the entire set, comment the following two lines. 
        '''
        real_data = real_data.iloc[:, 800:1201]
        synthetic_data = synthetic_data.iloc[:, 800:1201]
        Generate_quality_report(prediction_model_dim, real_data, synthetic_data, metadata_dict)
        Generate_diagnostic_report(prediction_model_dim, real_data, synthetic_data, metadata_dict)
    else:
        if prediction_model_dim.split('_')[-1] == 'susc':
            susc_data, _ = Select_susc_spectra(real_data)
            susc_data = susc_data.iloc[:, 800:1201]
            synthetic_data = synthetic_data.iloc[:, 800:1201]
            Generate_quality_report(prediction_model_dim, susc_data, synthetic_data, metadata_dict)
            Generate_diagnostic_report(prediction_model_dim, susc_data, synthetic_data, metadata_dict)
        elif prediction_model_dim.split('_')[-1] == 'res':
            res_data, _ = Select_res_spectra(real_data)
            res_data = res_data.iloc[:, 800:1201]
            synthetic_data = synthetic_data.iloc[:, 800:1201]
            Generate_quality_report(prediction_model_dim, res_data, synthetic_data, metadata_dict)
            Generate_diagnostic_report(prediction_model_dim, res_data, synthetic_data, metadata_dict)
        

def Argv_error_1():
    message = '''\nYou have to specify the file name.\n 
    Please follow the instructions and provide one file names.\n
    Ex.: '--python test_metrics.py predictiom_model'
    '''
    return print(message)

def Argv_error_2():
    message = '''\nYou can specify only one file.\n 
    Please follow the instructions and provide ONLY one file.\n
    Ex.: '--python test_metrics.py predictiom_model'
    '''
    return print(message)

def main():
    metadata_dict = Get_metadata_dict()
    
    # check if latent dimension parameter is provided
    if len(sys.argv)<2:
        Argv_error_1()
        sys.exit(1)
        
    # if one argument is provided
    elif len(sys.argv)==2:
        file_name = sys.argv[1]
        if file_name.split('_')[1] == 'vae':
            Generate_vae_reports(file_name, metadata_dict)
        elif file_name.split('_')[1] == 'cvae':
            Generate_cvae_reports(file_name, metadata_dict)
            
    else:       
        Argv_error_2()
    
if __name__ == '__main__':

    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    