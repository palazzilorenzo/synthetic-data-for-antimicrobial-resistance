| **Author**  | **Project** |
|:------------:|:-----------:|
| [**L. Palazzi**](https://github.com/palazzilorenzo) | **Synthetic data for mass spectrometry** |

![GitHub stars](https://img.shields.io/github/stars/palazzilorenzo/synthetic-data-for-mass-spectrometry?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/palazzilorenzo/synthetic-data-for-mass-spectrometry?label=Watch&style=social
)

# synthetic-data-for-mass-spectrometry

This package allows to generate synthetic tabular data for AMR using [Variational AutoEncoders](https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry/blob/main/extras/vae_structure.png) and [Conditional Variational AutoEncoders](https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry/blob/main/extras/cvae_structure.png).
This package provides a series of modules and notebook to build and train the models, generate new data starting from real ones, and test their quality and validity through the [SDMetrics](https://docs.sdv.dev/sdmetrics/) package. 

1. [Overview](#Overview)
2. [Contents](#Contents)
3. [Prerequisites](#Prerequisites)
4. [Installation](#Installation)
5. [Usage](#Usage)

## Overview

Antimicrobial Resistance (AMR) occurs when microbes like bacteria, viruses, and fungi develop the ability to withstand antimicrobial treatments (drugs used to treat infections), increasing the risk of disease spread, severe illness and death.  
Matrix-assisted laser desorption/ionization–time of flight (MALDI-TOF) mass spectrometry can be used to rapidly identify microbial species [2]. A recent work [4] has shown that machine learning-based method may be applied to MALDI-TOF spectra of bacteria to test antimicrobial susceptibility (or resistance), thus allowing more suitable treatment decisions in a clinical context. 
Since real data are often not enough to properly train a machine learning model (and thus to make correct predictions), the use of synthetic data is becoming increasingly important. 

This work will provide a model based on Variational AutoEncoders (VAEs) and Conditional AutoEncoders (CVAEs) to generate new synthetic sets of data, and will also test their goodness w.r.t. the original dataset using consolidated metrics. 

**Keywords**: synthetic data, mass spectrometry, MALDI-TOF, bacteria, bacteriology, antimicrobial resistance, AMR. 
 

### References 

[1] Bielow C, Aiche S, Andreotti S, Reinert K. MSSimulator: Simulation of Mass Spectrometry Data. Journal of Proteome Research 2011 10 (7), 2922-2929. doi: 10.1021/pr200155f. 

[2] Seng P, Drancourt M, Gouriet F, La Scola B, Fournier PE, Rolain JM, Raoult D. Ongoing revolution in bacteriology: routine identification of bacteria by matrix-assisted laser desorption ionization time-of-flight mass spectrometry. Clin Infect Dis. 2009 Aug 15;49(4):543-51. doi: 10.1086/600885. PMID: 19583519.  

[3] Sohn K, Lee H, Yan X. Learning Structured Output Representation using Deep Conditional Generative Models. Advances in Neural Information Processing Systems 28 (NIPS 2015). https://proceedings.neurips.cc/paper_files/paper/2015/file/8d55a249e6baa5c06772297520da2051-Paper.pdf.  

[4] Weis C, Cuénod A, Rieck B et al. Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning. Nat Med 28, 164–174 (2022). https://doi.org/10.1038/s41591-021-01619-9. 

## Contents
synthetic-data-for-mass-spectrometry is composed of a series of modules contained in [docs](https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry/tree/main/docs) and notebooks in [notebook](https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry/tree/main/notebook):
- modules define classes and methods useful to create the VAE/CVAE model.
- notebook contains jupyter notebooks that allow to train the model, generate new data and test their quality w.r.t. real data.

For a better description of each module:

| **Module**| **Description**|
|:---------:|:--------------:|
| class_cvae | define class for CVAE model |
| class_vae | define class for VAE model |
| create_cvae | contains the architecture of the CVAE model, i.e. encoder and decoder architecture |
| create_vae | contains the architecture of the VAE model, i.e. encoder and decoder architecture |
| sampling | contains method to sample z, the vector encoding a digit. |

For a better description of each notebook:

| **Notebook** | **Description** |
|:----------:|:---------------:|
| cvae_training | allows to build and train the CVAE model, saving weights and losses.	|
| cvae_prediction | allows to run predictions for CVAE model, i.e. generate new synthetic data.	|
| cvae_results | allows to test the quality of new data for CVAE model, also provides	graphical informations. |
| vae_training | allows to build and train the VAE model, saving weights and losses.	|
| vae_prediction | allows to run predictions for VAE model, i.e. generate new synthetic data.	|
| vae_results | allows to test the quality of new data for VAE model, also provides	graphical informations. |

## Prerequisites

Supported python: ![Python version](https://img.shields.io/badge/python--version-v3.11.5-blue)

First of all ensure to have the right python version installed.

This project use Tensorflow, keras, numpy: see the
[requirements](https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry/blob/main/requirements.txt) for more information.

## Installation
First, clone the git repository and change directory:

```bash
git clone https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry.git
cd synthetic-data-for-mass-spectrometry
```

Then, pip-install the requirements and run the setup script:
```bash
pip install -r requirements.txt
```
> :warning: Apple Silicon: ensure to have installed a proper [Conda env](https://github.com/conda-forge/miniforge), then install tensorflow as explained [here](https://developer.apple.com/metal/tensorflow-plugin/). Finally, install one by one the remaining dependencies using conda (if available) or pip.

## Usage

Once you have installed all the requirements, you can start to generate and test new set of data. 
First open Jupyter Notebook with:

```bash
   jupyter notebook 
```

Then from the Jupyter dashboard you can choose which notebook to run. For example, to simply generate new dataset from existing model, run vae_prediction.ipynb or cvae_prediction.ipynb according to the model one want to use, and save them with new name.

Follow instruction inside the notebooks to create new models, train them and run predictions.





