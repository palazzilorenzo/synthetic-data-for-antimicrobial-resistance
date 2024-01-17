# How to use
In this example a CVAE model with latent space dimension of 64 is built, trained and used to generate new set of data.
Some results are also present at the end to show how good new synthetic data are w.r.t. the real data.

## 1. Training step
Run:
```bash
python train_model.py cvae_64
```
to start the training. Something like this should be seen:

<p align="center">    
   <img src="./extras/imgs/training.png" width="500" height="300" />
 </p>

 The training stops when there is no further decrease on loss for ten consecutive epochs.
 The script will save model weights and history loss respectively to ./cvae/weights/ and ./cvae/history/ folders.

 ## 2. Prediction step
 Run:
 ```bash
python generate_data.py cvae_64
```
to generate new set of synthtic data. New data are saved in ./cvae/synthetic_data/ folder.
In this case the script will generate three different dataset:
- both susceptible and resistant spectra, saved as ```prediction_cvae_64```
- only susceptible spectra, saved as ```prediction_cvae_64_susc```
- only resistant spectra, saved as ```prediction_cvae_64_res```

**N.B.:** for VAE model generate_data.py will generate only one dataset, since we do not have the susceptibility label.

## 3. Testing step
Run:
```bash
python test_metrics.py prediction_cvae_64
```
to test the quality and validity of the synthetic dataset generated.
The reports containing the testing properties are saved in ./reports/quality and ./reports/diagnostic folders.

## 4. Results

```bash
 python graphics.py Show_history_loss cvae_64
```
<p align="center">    
   <img src="./extras/imgs/history_cvae_64.png" width="400" height="400" />
 </p>
 
---

```bash
 python graphics.py Show_synth_vs_real prediction_cvae_64
```
<p align="center">    
   <img src="./extras/imgs/synth_vs_real_cvae_64.png" width="600" height="400" />
 </p>
 
 ---

```bash
 python graphics.py Column_similarity prediction_cvae_64
```
<p align="center">    
   <img src="./extras/imgs/cvae_64_similarity_col_4601.png" width="600" height="400" />
 </p>
 
 ---

```bash
 python graphics.py Column_shapes cvae_64
```
<p align="center">    
   <img src="./extras/imgs/cvae_64_column_shapes.png" width="600" height="400" />
 </p>
 
 ---
 
```bash
 python graphics.py Column_pair_trends cvae_64
```
<p align="center">    
   <img src="./extras/imgs/cvae_64_column_pair_trends.png" width="700" height="700" />
 </p>
 
 ---
 
```bash
 python graphics.py Coverage cvae_64
```
<p align="center">    
   <img src="./extras/imgs/cvae_64_coverage.png" width="600" height="400" />
 </p>









