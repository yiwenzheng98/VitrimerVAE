# VitrimerVAE
Inverse Design of Vitrimeric Polymers by Molecular Dynamics and Generative Modeling: https://arxiv.org/abs/2312.03690

## ZINC
Datasets derived from ZINC15 database: 322K carboxylic acids, 625K epoxides, 1M vitrimers. The data will be made openly available at the time of publication.

## MD
Density-temperature profiles and calculated *T*<sub>g</sub> of 8,424 vitrimers for data generation and discovered vitrimers by VAE and Bayesian optimization. The data will be made openly available at the time of publication.

## Calibration
The codes are tested in the following environment:
 - Python (3.10.8)
 - RDKit (2022.09.1)
 - GPflow (2.5.2)
 - TensorFlow (2.9.1)

1. Train a GP model to calibrate MD-simulated *T*<sub>g</sub> against experimental *T*<sub>g</sub> of vitrimers:\
`python calibrate.py`
2. Compare the performance of different calibration models by LOOCV:\
`python linear.py`\
`python RBF.py`\
`python Tanimoto.py`

The vitrimer data will be made openly available at the time of publication.

## VAE
The VAE codes are based on https://github.com/wengong-jin/hgraph2graph/

The codes are tested in the following environment:
 - Python (3.6.13)
 - PyTorch (1.10.1)
 - RDKit (2019.03.4)
 - networkx (2.2)
 - Numpy (1.19.2)

To train the VAE and generate novel vitrimers:
1. Exrtact motif vocabulary from training set:\
`python get_vocab.py --ncpu 32`

2. Preprocess molecules into tensors:\
`python preprocess.py --ncpu 32`

3. Train the VAE on the unlabeled dataset:\
`python train_vae.py --savedir results`

4. Train the VAE on the labeled dataset (joint training):\
`python train_vae_prop.py --savedir results --model 9.model`

5. Evaluate the trained models:\
`python evaluate.py --savedir results`

6. Extract latent vectors:\
`python latent_vector.py --savedir results`

7. Search in the neighborhood of a known vitrimer:\
`python search.py --savedir results`

8. Interpolate bewteen two vitrimers:\
`python interpolate.py --savedir results --method spherical`

9. Inverse design by Bayesian optimization\
The Bayesian optimization codes require the customized Theano library from https://github.com/mkusner/grammarVAE#bayesian-optimization \
First generate initial data points:\
`python bo_initial.py --savedir results`\
Run Bayesian optimization:\
`python bo.py --savedir results --target 1`\
Collect 200 vitrimers with *T*<sub>g</sub> closest to target:\
`python bo_combine.py`

The training and test data will be made openly available at the time of publication.

## BO
This folder contains 10 proposed vitrimers with validated (by MD simulation and GP calibration) *T*<sub>g</sub> of three targets:

1. *T*<sub>g</sub> = 248 K
2. *T*<sub>g</sub> = 373 K
3. Maximum *T*<sub>g</sub>
