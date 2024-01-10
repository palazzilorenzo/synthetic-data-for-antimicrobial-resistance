#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py: 
    this module contains useful methods to access data, load models, train them and run predictions
"""

__author__ = ['Lorenzo Palazzi']
__email__ = ['lorenzo.palazzi@studio.unibo.it']

import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import json

from keras.utils import to_categorical
from sklearn.preprocessing import normalize
from keras.callbacks import EarlyStopping

from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table import DiagnosticReport

from docs.create_vae import Vae
from docs.create_cvae import Cvae

###############################
##                           ##
##  VARIATIONAL AUTOENCODER  ##
##                           ##
###############################

def Get_vae_training_data():
    '''
    
    Import training data and 
    prepare it to train the VAE model
    
    '''
    script_dir = os.getcwd()  # Script directory
    path_to_data = os.path.join(script_dir,'VAE/real_data/train/Escherichia_coli_VAE_train')
    extension = '.csv'
    
    train = pd.read_csv(path_to_data + extension, sep='\t', header=0)
    
    # Remove the 'Unnamed: 0' column
    train = train.drop(['Unnamed: 0'],  axis = 1)
    
    return train

def Get_vae_testing_data():
    '''
    
    Import testing data and 
    prepare it to test the VAE model
    
    '''
    work_dir = os.getcwd()  # Working directory
    path_to_data = os.path.join(work_dir,'VAE/real_data/test/Escherichia_coli_VAE_test')
    extension = '.csv'
    
    test = pd.read_csv(path_to_data + extension, sep='\t', header=0)
    
    # Remove the 'Unnamed: 0' column
    test = test.drop(['Unnamed: 0'],  axis = 1)
    
    # get the list of all columns that will be needed later
    col = test.columns.values.tolist()
    
    return test, col

def Get_vae_synthetic_data(file_name):
    '''
    
    Get synthetic data from VAE model
    as Pandas DataFrame
    
    '''
    work_dir = os.getcwd()  # Working directory
    path_to_data = os.path.join(work_dir,f'VAE/synthetic_data/{file_name}')
    extension = '.csv'
    
    synthetic_data = pd.read_csv(path_to_data + extension, sep='\t', header=0)
    
    # Remove the 'Unnamed: 0' column
    synthetic_data = synthetic_data.drop(['Unnamed: 0'],  axis = 1)
    
    return synthetic_data


def Load_vae_model(model_name, path_to_weights):
    latent_dim = int(model_name.split('_')[1])
    vae = Vae(latent_dim)
    vae.built = True
    vae.load_weights(path_to_weights + model_name + '.h5')
    return vae

def Train_vae(model_name, train_data):
    # This callback will stop the training when there is no improvement in
    # the loss for ten consecutive epochs.
    callback = EarlyStopping(monitor='loss', patience=10)
    
    work_dir = os.getcwd()  # Working directory
    file_path = os.path.join(work_dir,'VAE/weights/' + model_name + '.h5')
    if os.path.isfile(file_path):
        print(f'Updating weights for {model_name}')
        vae = Load_vae_model(model_name, os.path.join(work_dir,'VAE/weights/'))
        history_vae = vae.fit(train_data, epochs=150, batch_size=9, callbacks=[callback])
        print("Number of epochs run", len(history_vae.history['loss']))
        # save weights
        vae.save_weights(os.path.join(work_dir,'VAE/weights/') + f'{model_name}.h5')
        # save history on file
        with open(work_dir + '/VAE/history/' + f'history_{model_name}.json', 'w') as f:
            json.dump(history_vae.history, f)
    else:
        print(f'Starting training for {model_name}')
        latent_dim = int(model_name.split('_')[1])
        vae = Vae(latent_dim)
        history_vae = vae.fit(train_data, epochs=150, batch_size=9, callbacks=[callback])
        print("Number of epochs run", len(history_vae.history['loss']))
        # save weights
        vae.save_weights(os.path.join(work_dir,'VAE/weights/') + f'{model_name}.h5')
        # save history on file
        with open(work_dir + '/VAE/history/' + f'history_{model_name}.json', 'w') as f:
            json.dump(history_vae.history, f)

def Predict_vae(vae, model_name, data, path_to_weights, col_name):
    work_dir = os.getcwd()  # Working directory
    path_to_prediction = os.path.join(work_dir,'VAE/synthetic_data/')
    print('\n' + 'Run prediction with ' + model_name)
    # Prediction with VAE
    encoded = vae.encoder.predict(data)[-1]
    prediction_vae = vae.decoder.predict(encoded)
    prediction_vae = prediction_vae.reshape(52, 6000)
        
    # normalization
    prediction_vae = normalize(prediction_vae, norm='l1')
    df_prediction_vae = pd.DataFrame(prediction_vae, columns=col_name)

    # saving as a CSV file
    df_prediction_vae.to_csv(path_to_prediction + 'prediction_' + model_name + '.csv', sep ='\t')
    print('\n' + f'New data for {model_name} saved in:\n' + work_dir + '/VAE/synthetic_data')


###############################
##                           ##
##        CONDITIONAL        ##
##  VARIATIONAL AUTOENCODER  ##
##                           ##
###############################


def Get_cvae_training_data():
    '''
    
    Import training data and 
    prepare it to train the CVAE model
    
    '''
    work_dir = os.getcwd()  # Working directory
    path_to_train_x = os.path.join(work_dir,'CVAE/real_data/train/Escherichia_coli_CVAE_train_x')
    path_to_train_labels = os.path.join(work_dir,'CVAE/real_data/train/Escherichia_coli_CVAE_train_labels')
    extension = '.csv'

    train_x = pd.read_csv(path_to_train_x + extension, sep='\t', header=0)
    train_labels = pd.read_csv(path_to_train_labels + extension, sep='\t', header=0)
    train_x = train_x.drop(['Unnamed: 0'],  axis = 1)
    train_labels = train_labels.drop(['Unnamed: 0'],  axis = 1)
    train_labels = to_categorical(train_labels, num_classes=2)
    
    return train_x, train_labels


def Get_cvae_testing_data():
    '''
    
    Import testing data and 
    prepare it to test the CVAE model
    
    '''
    work_dir = os.getcwd()  # Working directory
    path_to_test_x = os.path.join(work_dir,'CVAE/real_data/test/Escherichia_coli_CVAE_test_x')
    path_to_test_labels = os.path.join(work_dir,'CVAE/real_data/test/Escherichia_coli_CVAE_test_labels')
    extension = '.csv'

    test_x = pd.read_csv(path_to_test_x + extension, sep='\t', header=0)
    test_labels = pd.read_csv(path_to_test_labels + extension, sep='\t', header=0)

    test = pd.concat([test_x, test_labels], axis=1)
    
    test_x = test_x.drop(['Unnamed: 0'],  axis = 1)

    test_labels = test_labels.drop(['Unnamed: 0'],  axis = 1)
    test_labels = to_categorical(test_labels, num_classes=2)

    test = test.drop(['Unnamed: 0'],  axis = 1)
    col = test.columns.values.tolist()
    
    return test_x, test_labels, test, col

def Get_cvae_synthetic_data(file_name):
    '''
    
    Get synthetic data from CVAE model
    as Pandas DataFrame
    
    '''
    work_dir = os.getcwd()  # Working directory
    path_to_data = os.path.join(work_dir,f'CVAE/synthetic_data/{file_name}')
    extension = '.csv'
    
    synthetic_data = pd.read_csv(path_to_data + extension, sep='\t', header=0)
    
    # Remove the 'Unnamed: 0' column
    synthetic_data = synthetic_data.drop(['Unnamed: 0', 'Ampicillin'],  axis = 1)
    
    return synthetic_data

def Load_cvae_model(model_name, path_to_weights):
    latent_dim = int(model_name.split('_')[1])
    cvae = Cvae(latent_dim)
    cvae.built = True
    cvae.load_weights(path_to_weights + model_name + '.h5')
    return cvae

def Train_cvae(model_name, train_x, train_labels):
    # This callback will stop the training when there is no improvement in
    # the loss for ten consecutive epochs.
    callback = EarlyStopping(monitor='loss', patience=10)
    
    work_dir = os.getcwd()  # Working directory
    file_path = os.path.join(work_dir,'CVAE/weights/' + model_name + '.h5')
    if os.path.isfile(file_path):
        print(f'Updating weights for {model_name}')
        cvae = Load_cvae_model(model_name, os.path.join(work_dir,'CVAE/weights/'))
        history_cvae = cvae.fit(train_x, train_labels, batch_size=6, epochs=120, callbacks=[callback])
        print("Number of epochs run", len(history_cvae.history['loss']))
        # save weights
        cvae.save_weights(os.path.join(work_dir,'CVAE/weights/') + f'{model_name}.h5')
        # save history on file
        with open(work_dir + '/CVAE/history/' + f'history_{model_name}.json', 'w') as f:
            json.dump(history_cvae.history, f)
    else:
        print(f'Starting training for {model_name}')
        latent_dim = int(model_name.split('_')[1])
        cvae = Cvae(latent_dim)
        history_cvae = cvae.fit(train_x, train_labels, batch_size=6, epochs=120, callbacks=[callback])
        print("Number of epochs run", len(history_cvae.history['loss']))
        # save weights
        cvae.save_weights(os.path.join(work_dir,'CVAE/weights/') + f'{model_name}.h5')
        # save history on file
        with open(work_dir + '/CVAE/history/' + f'history_{model_name}.json', 'w') as f:
            json.dump(history_cvae.history, f)

def Predict_cvae(cvae, model_name, x, labels, col_name):
    work_dir = os.getcwd()  # Working directory
    synthetic_data_path = os.path.join(work_dir,'CVAE/synthetic_data/')
    file_name = synthetic_data_path + 'prediction_' + model_name + '.csv'
    print('\n' + 'Generate resistant and susceptible spectra with ' + model_name)
    
    # prediction
    encoded = cvae.encoder.predict([x, labels]) # encode data
    z_sample = encoded[2]
    prediction_cvae = cvae.decoder.predict([z_sample, labels]) # decode data
    prediction_cvae = prediction_cvae[:,:,0] 
    
    # separate intensity and label values to normalize
    prediction_cvae_x = prediction_cvae[:,:6000] # intensity
    prediction_cvae_labels = prediction_cvae[:,-2:] # labels
    prediction_cvae_labels = np.argmax(prediction_cvae_labels, axis=1) # inverse of 'to_categorical'
    prediction_cvae_labels = np.expand_dims(prediction_cvae_labels, -1).astype("int32")
    # normalization
    prediction_cvae_x_norm = normalize(prediction_cvae_x, norm='l1')
    # reunite normalized spectra with labels
    prediction_cvae_norm = np.hstack((prediction_cvae_x_norm, prediction_cvae_labels))
    df_decoded_norm = pd.DataFrame(prediction_cvae_norm, columns=col_name)
    # saving as a CSV file
    df_decoded_norm.to_csv(file_name, sep ='\t')
    print('\n' + f'New resistant and susceptible spectra for {model_name} saved in:\n' + synthetic_data_path)

def Predict_cvae_susc(cvae, model_name, x, labels, col_name):
    work_dir = os.getcwd()  # Working directory
    synthetic_data_path = os.path.join(work_dir,'CVAE/synthetic_data/')
    file_name = synthetic_data_path + 'prediction_' + model_name + '_susc' + '.csv'
    print('\n' + 'Generate only susceptible spectra with ' + model_name)
    
    # prediction
    encoded = cvae.encoder.predict([x, labels]) # encode data
    z_sample = encoded[2]
    prediction_cvae = cvae.decoder.predict([z_sample, labels]) # decode data
    prediction_cvae = prediction_cvae[:,:,0] 
    
    # separate intensity and label values to normalize
    prediction_cvae_x = prediction_cvae[:,:6000] # intensity
    prediction_cvae_labels = prediction_cvae[:,-2:] # labels
    prediction_cvae_labels = np.argmax(prediction_cvae_labels, axis=1) # inverse of 'to_categorical'
    prediction_cvae_labels = np.expand_dims(prediction_cvae_labels, -1).astype("int32")
    # normalization
    prediction_cvae_x_norm = normalize(prediction_cvae_x, norm='l1')
    # reunite normalized spectra with labels
    prediction_cvae_norm = np.hstack((prediction_cvae_x_norm, prediction_cvae_labels))
    df_decoded_norm = pd.DataFrame(prediction_cvae_norm, columns=col_name)
    # saving as CSV file
    df_decoded_norm.to_csv(file_name, sep ='\t')
    print('\n' + f'New susceptible spectra for {model_name} saved in:\n' + synthetic_data_path)
    
def Predict_cvae_res(cvae, model_name, x, labels, col_name):
    work_dir = os.getcwd()  # Working directory
    synthetic_data_path = os.path.join(work_dir,'CVAE/synthetic_data/')
    file_name = synthetic_data_path + 'prediction_' + model_name + '_res' + '.csv'
    print('\n' + 'Generate only resistant spectra with ' + model_name)
    
    # prediction
    encoded = cvae.encoder.predict([x, labels]) # encode data
    z_sample = encoded[2]
    prediction_cvae = cvae.decoder.predict([z_sample, labels]) # decode data
    prediction_cvae = prediction_cvae[:,:,0] 
    
    # separate intensity and label values to normalize
    prediction_cvae_x = prediction_cvae[:,:6000] # intensity
    prediction_cvae_labels = prediction_cvae[:,-2:] # labels
    prediction_cvae_labels = np.argmax(prediction_cvae_labels, axis=1) # inverse of 'to_categorical'
    prediction_cvae_labels = np.expand_dims(prediction_cvae_labels, -1).astype("int32")
    # normalization
    prediction_cvae_x_norm = normalize(prediction_cvae_x, norm='l1')
    # reunite normalized spectra with labels
    prediction_cvae_norm = np.hstack((prediction_cvae_x_norm, prediction_cvae_labels))
    df_decoded_norm = pd.DataFrame(prediction_cvae_norm, columns=col_name)
    # saving as CSV file
    df_decoded_norm.to_csv(file_name, sep ='\t')
    print('\n' + f'New resistant spectra for {model_name} saved in:\n' + synthetic_data_path)


###############################
##                           ##
##          REPORTS          ##
##                           ##
###############################

def Get_metadata_dict():
    ''' 
    To reduce computational time we'll test only the range 4400-5600.
    If you want to test the entire set, replace 'my_metadata_file_red.json' with 'my_metadata_file.json'. 
    '''
    work_dir = os.getcwd()  # Working directory    
    metadata_path = os.path.join(work_dir, 'metadata_json/my_metadata_file_red.json')
    
    with open(metadata_path) as f:
        metadata_dict = json.load(f)
        
    return metadata_dict

def Generate_quality_report(file_name, real_data, synthetic_data, metadata_dict):
    model_name = file_name.split('_')[1] + '_' + file_name.split('_')[2]
    work_dir = os.getcwd()  # Working directory
    quality_reports_path = os.path.join(work_dir,f'reports/quality/quality_report_{model_name}.pkl')

    # generate quality report
    report = QualityReport()
    report.generate(real_data, synthetic_data, metadata_dict)
    # save report
    report.save(filepath=quality_reports_path)
    print(f'Quality report saved: {quality_reports_path}')
    
def Generate_diagnostic_report(file_name, real_data, synthetic_data, metadata_dict):
    model_name = file_name.split('_')[1] + '_' + file_name.split('_')[2]
    work_dir = os.getcwd()  # Working directory
    diagnostic_reports_path = os.path.join(work_dir,f'reports/diagnostic/diagnostic_report_{model_name}.pkl')

    # generate quality report
    report = DiagnosticReport()
    report.generate(real_data, synthetic_data, metadata_dict)
    # save report
    report.save(filepath=diagnostic_reports_path)
    print(f'Diagnostic report saved: {diagnostic_reports_path}')

