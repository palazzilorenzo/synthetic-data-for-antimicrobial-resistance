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

from docs.utils import Get_vae_testing_data, Get_vae_synthetic_data, Get_metadata_dict
from docs.utils import Get_cvae_testing_data, Get_cvae_synthetic_data
from docs.utils import Generate_quality_report, Generate_diagnostic_report


def Generate_vae_reports(file_name, metadata_dict):
    real_data, _ = Get_vae_testing_data()
    synthetic_data = Get_vae_synthetic_data(file_name)
    ''' 
    To reduce computational time we'll test only the range 4400-5600.
    If you want to test the entire set, comment the following two lines. 
    '''
    real_data = real_data.iloc[:, 800:1201]
    synthetic_data = synthetic_data.iloc[:, 800:1201]
    
    Generate_quality_report(file_name, real_data, synthetic_data, metadata_dict)
    Generate_diagnostic_report(file_name, real_data, synthetic_data, metadata_dict)
    
def Generate_cvae_reports(file_name, metadata_dict):
    real_data, _, _, _ = Get_cvae_testing_data()
    synthetic_data = Get_cvae_synthetic_data(file_name)
    ''' 
    To reduce computational time we'll test only the range 4400-5600.
    If you want to test the entire set, comment the following two lines. 
    '''
    real_data = real_data.iloc[:, 800:1201]
    synthetic_data = synthetic_data.iloc[:, 800:1201]
    
    if len(file_name.split('_')) == 3:
        Generate_quality_report(file_name, real_data, synthetic_data, metadata_dict)
        Generate_diagnostic_report(file_name, real_data, synthetic_data, metadata_dict)
    else:
        if file_name.split('_')[-1] == 'susc':
            susc_data = real_data.loc[real_data['Ampicillin'] == 1] # select only resistant spectra
            susc_data = susc_data.drop(['Ampicillin'], axis = 1) # intensity values
            Generate_quality_report(file_name, susc_data, synthetic_data, metadata_dict)
            Generate_diagnostic_report(file_name, susc_data, synthetic_data, metadata_dict)
        elif file_name.split('_')[-1] == 'res':
            res_data = real_data.loc[real_data['Ampicillin'] == 0] # select only resistant spectra
            res_data = res_data.drop(['Ampicillin'], axis = 1) # intensity values
            Generate_quality_report(file_name, res_data, synthetic_data, metadata_dict)
            Generate_diagnostic_report(file_name, res_data, synthetic_data, metadata_dict)
        

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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    