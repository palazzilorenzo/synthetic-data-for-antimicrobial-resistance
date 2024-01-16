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
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table import DiagnosticReport

from docs.create_vae import Vae
from docs.create_cvae import Cvae

##############################
##                          ##
##   GET DATA,              ##
##   LOAD AND TRAIN MODEL,  ##
##   RUN PREDICTION         ##
##                          ##
##############################

def Get_data(model_name):
    '''
    
    Import training data as Pandas DataFrame
    
    '''
    work_dir = os.getcwd()  # Working directory
    extension = '.txt'
    
    if model_name == 'vae':
        path_to_data = os.path.join(work_dir,f'{model_name}/real_data/Escherichia_coli_{model_name.upper()}')
        data = pd.read_csv(path_to_data + extension, sep='\t', header=0)
        # Remove the 'Unnamed: 0' and 'Meropenem' column that will be obsolete for our purpose
        data = data.drop(['Unnamed: 0', 'Meropenem'],  axis = 1)
        
        return data
    
    elif model_name == 'cvae':
        path_to_data = os.path.join(work_dir,f'{model_name}/real_data/Escherichia_coli_{model_name.upper()}')
        data = pd.read_csv(path_to_data + extension, sep='\t', header=0)
        # Remove the 'Unnamed: 0' column to select only the intensity values and susceptibility
        data = data.drop(['Unnamed: 0'],  axis = 1)
        
        # Replace 'R' and 'S' with 0 and 1
        data['Ampicillin'] = data['Ampicillin'].replace('R', 0)
        data['Ampicillin'] = data['Ampicillin'].replace('S', 1)
        X = data.drop(['Ampicillin'], axis = 1) # dataframe containing only intensity values
        labels = pd.DataFrame(data['Ampicillin']) # dataframe containing only susceptibility information
        
        return X, labels

def Split_data(model_name):
    '''
    Split data into train and test set
    '''
    extension = '.csv'
    if model_name == 'vae':
        data = Get_data(model_name)
        train, test = train_test_split(data, random_state=42)
        
        path_to_train = f'{model_name}/real_data/train/Escherichia_coli_{model_name.upper()}_train'
        train.to_csv(path_to_train + extension, sep ='\t')
        
        path_to_test = f'{model_name}/real_data/test/Escherichia_coli_{model_name.upper()}_test'
        test.to_csv(path_to_test + extension, sep ='\t')
    elif model_name == 'cvae':
        data, labels = Get_data(model_name)
        # Split the data, stratifying by label
        train_x, test_x, train_labels, test_labels = train_test_split(data, labels,
                                                                      stratify=labels, 
                                                                      random_state=42)
        # saving as a CSV file
        path_to_train_x = f'{model_name}/real_data/train/Escherichia_coli_{model_name.upper()}_train_x'
        train_x.to_csv(path_to_train_x + extension, sep ='\t')
        
        path_to_train_labels = f'{model_name}/real_data/train/Escherichia_coli_{model_name.upper()}_train_labels'
        train_labels.to_csv(path_to_train_labels + extension, sep ='\t')
        
        path_to_test_x = f'{model_name}/real_data/test/Escherichia_coli_{model_name.upper()}_test_x'
        test_x.to_csv(path_to_test_x + extension, sep ='\t')
        
        path_to_test_labels = f'{model_name}/real_data/test/Escherichia_coli_{model_name.upper()}_test_labels'
        train_labels.to_csv(path_to_test_labels + extension, sep ='\t')

def Get_training_data(model_name):
    '''
    
    Import training data and 
    prepare it to train the model
    
    '''
    work_dir = os.getcwd()  # Working directory
    extension = '.csv'
    
    if model_name == 'vae':
        
        path_to_train = os.path.join(work_dir,f'{model_name}/real_data/train/Escherichia_coli_{model_name.upper()}_train')
        
        train = pd.read_csv(path_to_train + extension, sep='\t', header=0)
        
        # Remove the 'Unnamed: 0' column
        train = train.drop(['Unnamed: 0'],  axis = 1)
        
        return train
    
    elif model_name == 'cvae':
        
        path_to_train_x = os.path.join(work_dir,f'{model_name}/real_data/train/Escherichia_coli_{model_name.upper()}_train_x')
        path_to_train_labels = os.path.join(work_dir,f'{model_name}/real_data/train/Escherichia_coli_{model_name.upper()}_train_labels')
        
        train_x = pd.read_csv(path_to_train_x + extension, sep='\t', header=0)
        train_labels = pd.read_csv(path_to_train_labels + extension, sep='\t', header=0)
        train_x = train_x.drop(['Unnamed: 0'],  axis = 1)
        train_labels = train_labels.drop(['Unnamed: 0'],  axis = 1)
        train_labels = to_categorical(train_labels, num_classes=2)
        
        return train_x, train_labels
    
def Get_testing_data(model_name):
    '''
    
    Import testing data and 
    prepare it to test the model
    
    '''
    work_dir = os.getcwd()  # Working directory
    extension = '.csv'
    if model_name == 'vae':
        
        path_to_data = os.path.join(work_dir,f'{model_name}/real_data/test/Escherichia_coli_{model_name.upper()}_test')
        test = pd.read_csv(path_to_data + extension, sep='\t', header=0)
        
        # Remove the 'Unnamed: 0' column
        test = test.drop(['Unnamed: 0'],  axis = 1)
        
        # get the list of all columns that will be needed later
        col = test.columns.values.tolist()
        
        return test, col
    elif model_name == 'cvae':
        path_to_test_x = os.path.join(work_dir,f'{model_name}/real_data/test/Escherichia_coli_{model_name.upper()}_test_x')
        path_to_test_labels = os.path.join(work_dir,f'{model_name}/real_data/test/Escherichia_coli_{model_name.upper()}_test_labels')
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

def Select_susc_spectra(data):
            data_susc = data.loc[data['Ampicillin'] == 1] # select only susceptible spectra
            x_susc = data_susc.drop(['Ampicillin'], axis = 1) # intensity values
            labels_susc = pd.DataFrame(data_susc['Ampicillin']) # labels
            labels_susc = to_categorical(labels_susc, num_classes=2)
            return x_susc, labels_susc
        
def Select_res_spectra(data):
            data_res = data.loc[data['Ampicillin'] == 1] # select only susceptible spectra
            x_res = data_res.drop(['Ampicillin'], axis = 1) # intensity values
            labels_res = pd.DataFrame(data_res['Ampicillin']) # labels
            labels_res = to_categorical(labels_res, num_classes=2)
            return x_res, labels_res

def Get_synthetic_data(file_name):
    '''
    
    Get synthetic data and return it as Pandas DataFrame.
    
    Be sure that 'file_name' is in 'prediction_model_dim' format.
    
    '''
    work_dir = os.getcwd()  # Working directory
    model = file_name.split('_')[1]
    path_to_data = os.path.join(work_dir,f'{model}/synthetic_data/{file_name}')
    extension = '.csv'
    
    synthetic_data = pd.read_csv(path_to_data + extension, sep='\t', header=0)
    
    # Remove the 'Unnamed: 0' column
    synthetic_data = synthetic_data.drop(['Unnamed: 0'],  axis = 1)
    
    return synthetic_data


def Load_model(model_dim, path_to_weights):
    if model_dim.split('_')[0] == 'vae':
        latent_dim = int(model_dim.split('_')[1])
        vae = Vae(latent_dim)
        vae.built = True
        vae.load_weights(path_to_weights + model_dim + '.h5')
        return vae
    elif model_dim.split('_')[0] == 'cvae':
        latent_dim = int(model_dim.split('_')[1])
        cvae = Cvae(latent_dim)
        cvae.built = True
        cvae.load_weights(path_to_weights + f'{model_dim}.h5')
        return cvae

def Train_model(model_dim, train_data, labels=None):
    # This callback will stop the training when there is no improvement in
    # the loss for ten consecutive epochs.
    callback = EarlyStopping(monitor='loss', patience=10)
    
    work_dir = os.getcwd() # Working directory
    model_name = model_dim.split('_')[0]
    file_path = os.path.join(work_dir,f'{model_name}/weights/{model_dim}.h5')
    if os.path.isfile(file_path):
        print(f'Updating weights for {model_dim}')
        model = Load_model(model_dim, os.path.join(work_dir,f'{model_name}/weights/'))
        if model_name=='vae':
            history_loss = model.fit(train_data, epochs=150, batch_size=9, callbacks=[callback])
            print("Number of epochs run", len(history_loss.history['loss']))
        elif model_name=='cvae':
            history_loss = model.fit(train_data, labels, epochs=150, batch_size=6, callbacks=[callback])
            print("Number of epochs run", len(history_loss.history['loss']))
        # save weights
        model.save_weights(os.path.join(work_dir,f'{model_name}/weights/') + f'{model_dim}.h5')
        # save history on file
        with open(work_dir + f'/{model_name}/history/' + f'history_{model_dim}.json', 'w') as f:
            json.dump(history_loss.history, f)
    else:
        print(f'Starting training for {model_dim}')
        latent_dim = int(model_dim.split('_')[1])
        if model_name=='vae':
            model = Vae(latent_dim)
            history_loss = model.fit(train_data, epochs=150, batch_size=9, callbacks=[callback])
            print("Number of epochs run", len(history_loss.history['loss']))
        elif model_name=='cvae':
            model = Cvae(latent_dim)
            history_loss = model.fit(train_data, labels, epochs=150, batch_size=6, callbacks=[callback])
            print("Number of epochs run", len(history_loss.history['loss']))
        # save weights
        model.save_weights(os.path.join(work_dir, f'{model_name}/weights/') + f'{model_dim}.h5')
        # save history on file
        with open(work_dir + f'/{model_name}/history/' + f'history_{model_dim}.json', 'w') as f:
            json.dump(history_loss.history, f)

def Predict_data(model, model_dim, data, col_name, labels=None, susc=None):
    model_name = model_dim.split('_')[0]
    work_dir = os.getcwd()  # Working directory
    path_to_prediction = os.path.join(work_dir,f'{model_name}/synthetic_data/')
    print('\n' + 'Run prediction with ' + model_dim)
    # Prediction with VAE
    if model_name=='vae':   
        encoded = model.encoder.predict(data)[-1]
        prediction = model.decoder.predict(encoded)
        prediction = prediction.reshape(52, 6000)
        # normalization
        prediction = normalize(prediction, norm='l1')
    elif model_name=='cvae':
        # prediction
        encoded = model.encoder.predict([data, labels]) # encode data
        z_sample = encoded[2]
        prediction = model.decoder.predict([z_sample, labels]) # decode data
        prediction = prediction[:,:,0]
        # separate intensity and label values to normalize
        prediction_x = prediction[:,:6000] # intensity
        prediction_labels = prediction[:,-2:] # labels
        prediction_labels = np.argmax(prediction_labels, axis=1) # inverse of 'to_categorical'
        prediction_labels = np.expand_dims(prediction_labels, -1).astype("int32")
        # normalization
        prediction_x = normalize(prediction_x, norm='l1')
        # reunite normalized spectra with labels
        prediction = np.hstack((prediction_x, prediction_labels))

    prediction = pd.DataFrame(prediction, columns=col_name)
    if susc == 1:
        prediction.to_csv(path_to_prediction + f'prediction_{model_dim}_susc' + '.csv', sep ='\t')
    elif susc == 0:
        prediction.to_csv(path_to_prediction + f'prediction_{model_dim}_res' + '.csv', sep ='\t')
    else:
        # saving as a CSV file
        prediction.to_csv(path_to_prediction + f'prediction_{model_dim}' + '.csv', sep ='\t')
    print('\n' + f'New data for {model_dim} saved in:\n' + work_dir + f'/{model_name}/synthetic_data')
    

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

