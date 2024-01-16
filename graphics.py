#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graphics.py: 
    this module allows to show and save plots and results.
"""

__author__ = ['Lorenzo Palazzi']
__email__ = ['lorenzo.palazzi@studio.unibo.it']

import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import json

from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from sdmetrics.visualization import get_column_plot

from docs.utils import Get_testing_data, Get_synthetic_data

def Show_history_loss(model):
    model_name = model.split('_')[0]
    file_path = f'{model_name}/history/history_{model}.json'
    with open(file_path) as f:
        history = json.load(f)
        
    # Visualise loss for CVAE training
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    ax.plot(history['loss'])
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    ax.set_title(f'{model.upper()}')
    
    fig_path = f'results/history_{model}.png'
    print(f'Figure saved in results/ as history_{model}.png')
    plt.savefig(fig_path)
    plt.show()
    
def Show_synth_vs_real(file_name):
    model_name = file_name.split('_')[1]
    dim = file_name.split('_')[-1]
    if model_name == 'vae':
        real_data, _ = Get_testing_data(model_name)
        real_data = real_data.to_numpy()
        synth_data = Get_synthetic_data(file_name)
        synth_data = synth_data.to_numpy()
        
        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        # synthetic spectrum
        ax[0].plot(synth_data[0])
        ax[0].set_ylim(top=0.01)
        ax[0].set_title('Synthetic spectrum')
        ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # real spectrum
        ax[1].plot(real_data[0])
        ax[1].set_ylim(top=0.01)
        ax[1].set_title('Real spectrum')
        ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        fig_path = f'results/synth_vs_real_{model_name}_{dim}.png'
        print(f'Figure saved in results/ as synth_vs_real_{model_name}_{dim}.png')
        fig.savefig(fig_path)
        plt.show()
        
    elif model_name == 'cvae':
        real_data, _, _, _ = Get_testing_data(model_name)
        real_data = real_data.to_numpy()
        synth_data = Get_synthetic_data(file_name)
        # Remove the 'Ampicillin' column
        synth_data = synth_data.drop(['Ampicillin'],  axis = 1)
        synth_data = synth_data.to_numpy()
        
        fig, ax = plt.subplots(1, 2, figsize=(8, 5))
        # synthetic spectrum
        ax[0].plot(synth_data[0])
        ax[0].set_ylim(top=0.01)
        ax[0].set_title('Synthetic spectrum')
        ax[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # real spectrum
        ax[1].plot(real_data[0])
        ax[1].set_ylim(top=0.01)
        ax[1].set_title('Real spectrum')
        ax[1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        
        fig_path = f'results/synth_vs_real_{model_name}_{dim}.png'
        print(f'Figure saved in results/ as synth_vs_real_{model_name}_{dim}.png')
        fig.savefig(fig_path)
        plt.show()

def Column_similarity(file_name, column_name=4601):
    ''' Visualize the similarity between real and synthetic data
        (in this case we selected randomly the column '4601' as default).
     '''
    model_name = file_name.split('_')[1]
    dim = file_name.split('_')[-1]
    if model_name == 'vae':
        real_data, _ = Get_testing_data(model_name)
        synth_data = Get_synthetic_data(file_name)
        fig = get_column_plot(
            real_data=real_data,
            synthetic_data=synth_data,
            column_name=f'{column_name}',
            plot_type='distplot'
        )
        
        fig_path = f'results/{model_name}_{dim}_similarity_col_{column_name}.png'
        print(f'Figure saved in results/ as {model_name}_{dim}_similarity_col_{column_name}.png')
        fig.write_image(fig_path, scale=2.0)
        
        fig.show()   
    elif model_name == 'cvae':
        real_data, _, _, _ = Get_testing_data(model_name)
        synth_data = Get_synthetic_data(file_name)
        # Remove the 'Ampicillin' column
        synth_data = synth_data.drop(['Ampicillin'],  axis = 1)
        fig = get_column_plot(
            real_data=real_data,
            synthetic_data=synth_data,
            column_name=f'{column_name}',
            plot_type='distplot'
        )
        
        fig_path = f'results/{model_name}_{dim}_similarity_col_{column_name}.png'
        print(f'Figure saved in results/ as {model_name}_{dim}_similarity_col_{column_name}.png')
        fig.write_image(fig_path, scale=2.0)
        
        fig.show()

def Column_shapes(model):
    report = QualityReport.load(f'reports/quality/quality_report_{model}.pkl')
    fig = report.get_visualization(property_name='Column Shapes')
    fig.write_image('results/{model}_column_shapes.png', scale=2.0)
    print(f'Figure saved in results/ as {model}_column_shapes.png')
    fig.show()
    
def Column_pair_trends(model):
    report = QualityReport.load(f'reports/quality/quality_report_{model}.pkl')
    fig = report.get_visualization(property_name='Column Pair Trends')
    fig.write_image(f'results/{model}_column_pair_trends.png', scale=2.0)
    print(f'Figure saved in results/ as {model}_column_pair_trends.png')
    fig.show()

def Coverage(model):
    report = DiagnosticReport.load(f'reports/diagnostic/diagnostic_report_{model}.pkl')
    fig = report.get_visualization(property_name='Coverage')

    fig.write_image(f'results/{model}_coverage.png', scale=2.0)
    print(f'Figure saved in results/ as {model}_coverage.png')
    fig.show()

def Args_message():
    print(''''
          Please provide the right number of arguments. \n
          Follow the instruction and try again.
          Ex.: --python graphics.py Show_history_loss cvae_64
          ''')

def main():
    args = sys.argv
    if len(args) == 3:
        globals()[args[1]](args[2])
    elif len(args) == 4:
        globals()[args[1]](args[2], args[3])
    else:
        Args_message()
    
if __name__ == '__main__':
    main()
