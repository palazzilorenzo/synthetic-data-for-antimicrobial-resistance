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
synthetic-data-for-mass-spectrometry is composed of a series of modules contained in [docs](https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry/tree/main/docs) and [VAE](https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry/tree/main/VAE)/[CVAE](https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry/tree/main/CVAE) folders, according to the model one want to use:

- docs contains modules defining classes and methods useful to create the VAE/CVAE model.
- VAE/CVAE contains modules that allow to train the model and generate new data, and save all the informations.

Some examples are also present in the [notebook](https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry/tree/main/notebook) folder.

For a better description of each module:

| **Module**| **Description**|
|:---------:|:--------------:|
| class_cvae | define class for CVAE model |
| class_vae | define class for VAE model |
| create_cvae | contains the architecture of the CVAE model, i.e. encoder and decoder architecture |
| create_vae | contains the architecture of the VAE model, i.e. encoder and decoder architecture |
| sampling | contains method to sample z, the vector encoding a digit. |
| utils | this module contains useful methods to access data, load models, train them and run predictions |

For a better description of each notebook:

| **VAE/CVAE** | **Description** |
|:----------:|:---------------:|
| train_vae/cvae | allows to build and train the VAE/CVAE model, saving weights and losses.	|
| prediction_vae/cvae | allows to run predictions with existing VAE/CVAE model, i.e. generate new synthetic data.	|

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

Once you have installed all the requirements, you can start by training your model/s. 
#### Step 1:
According to the model you want to use (cvae or vae):
```bash
   python CVAE/train_cvae.py model_dim
```
or
```bash
   python VAE/train_vae.py model_dim
```
where 
* ```model``` is the name of the model you want to use, i.e. cvae or vae (required) ;
* ```dim``` is the dimension of the latent space, i.e. the space in which data are represented after encoding process (required).
>:warning: ```model``` and ```dim``` must be separated by the '_' character.

Example: 
```bash
   python CVAE/train_cvae.py cvae_64
```
or
```bash
   python VAE/train_vae.py vae_64
```
You can train more than one model in the same execution by separate them by one space.
>:warning: ```model``` must be the same, ```dim``` can change.

Example:
```bash
   python VAE/train_vae.py vae_32 vae_64
```

#### Step 2:
Now that you have built and trained your model/s, you can generate new set of synthetic data.
```bash
   python CVAE/prediction_cvae.py model_dim
```
or
```bash
   python VAE/prediction_vae.py model_dim
```
Same rules as before for ```model_dim```.

Synhtetic data are saved as ```prediction_model_dim``` format.

#### Step 3:
You can now test the quality and validity of your new data w.r.t. the real ones by running:
or
```bash
   python test_metrics.py file_name
```
where
* ```file_name``` is the name of the file containing the synthetic data you want to test.
>:warning: ```file_name``` must not contain the file extension.

Examle:
```bash
   python test_metrics.py prediction_cvae_64
```

## Author
* <img src="https://avatars.githubusercontent.com/u/135356553?v=4" width="25px;"/> **Lorenzo Palazzi** [git](https://github.com/palazzilorenzo)

### Citation

If you have found `synthetic-data-for-mass-spectrometry` helpful in your research, please
consider citing this project

```BibTeX
@misc{Synthetic data for mass spectrometry,
  author = {Palazzi, Lorenzo},
  title = {Synthetic data for mass spectrometry},
  year = {2024},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/palazzilorenzo/synthetic-data-for-mass-spectrometry}},
}
```





