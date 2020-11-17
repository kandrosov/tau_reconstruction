import ROOT as R
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf as pp
from tensorflow import keras

from training import training, make_generator
from mymodel import MyModel
from plotting import plot_metrics, plt_conf_dm, accuracy_calc
from evaluation import evaluation

n_tau    = 100
n_pf     = 100
n_fe     = 6
n_counts = 2
n_epoch  = 10
n_steps  = 10
entry_start = 0
entry_stop  = 10000
entry_start_val = entry_stop+1
entry_stop_val  = entry_stop+1001

model = MyModel()

R.gInterpreter.ProcessLine('#include "DataLoader.h"')

# total number of events in the dataset = 14215297
generator, n_batches = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start, entry_stop, n_tau, n_pf, n_fe, n_counts)
generator_xyz, n_batches = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start, entry_stop, n_tau, n_pf, n_fe, n_counts, z = True)
generator_val, n_batches_val = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start_val, entry_stop_val, n_tau, n_pf, n_fe, n_counts)    

history = training(model, generator, generator_val, n_tau, n_pf, n_fe, n_counts, n_epoch, n_batches, n_steps) # trains the model

plot_metrics(history, n_epoch) # Plots the loss and accuracy curves

conf_dm_mat, conf_dm_mat_old = evaluation(model,generator_xyz,n_steps, n_tau) # creates the confusion matrices

plt_conf_dm(conf_dm_mat)                 # Configuration matrix of decay modes (truth vs. predicted)
plt_conf_dm(conf_dm_mat_old, old = True) # Confusion matrix to compare predicted decay modes to old reconstruction method

accuracy = accuracy_calc(conf_dm_mat) # Accuracy calculation of main decay modes
print('Accuracy compared to true decay modes: \n',accuracy,'\n')
