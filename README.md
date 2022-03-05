# Neural Optimal Feedback Control

This repository is the official implementation of [Neural optimal feedback control with local learning rules](https://papers.nips.cc/paper/2021/hash/88591b4d3219675bdeb33584b755f680-Abstract.html), which has been published as part of Advances in Neural Information Processing Systems 35 (NeurIPS 2021).

![Image of Bio-OFC circuit and learning rules](https://github.com/j-friedrich/neuralOFC/blob/master/fig/fig1_circuit_and_learning_rules.png)

## Requirements

The scripts to reproduce all figures of the paper require a typical numerical/scientific Python installation that includes the following

- python
- matplotlib
- numpy
- scipy

We used optuna to perform hyperparameter optimization. The optimal hyperparameters are included in the results directory, thus the following is optional

- optuna 

To install requirements (using [conda](https://www.anaconda.com/products/individual)) execute:

```setup
conda env create -f environment.yml
conda activate neuralOFC
```

## Training

Pre-trained models and optimal hyperparameters are included in the results directory of this repository.
To nevertheless re-train the models and perform the hyperparameter optimization, which requires optuna (`conda install optuna`), (re)move the files from the results directory and run these commands:

```train
python fig3_delay.py
sh hyperopt.sh
```

## Evaluation

To reproduce the figures, run for each script

```fig
python <name_of_fig_script.py>
```
The figures will be saved in the fig directory of this repository.

## Pre-trained Models

The pre-trained models are included in the results directory of this repository.
