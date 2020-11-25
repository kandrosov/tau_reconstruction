import ROOT as R
import matplotlib.pyplot as plt
import numpy as np

R.gInterpreter.ProcessLine('#include "DataLoader.h"')

from training import training
from mymodel import * #gives us all n_tau, n_pf, ....
from plotting import plot_metrics, plt_conf_dm
from evaluation import evaluation, accuracy_calc


history = training() # trains the model

plot_metrics(history) # Plots the loss and accuracy curves for trainset and validationset

conf_dm_mat, conf_dm_mat_old = evaluation() # creates the confusion matrices

print('\nEvaluation is finished.\n')

plt_conf_dm(conf_dm_mat)  # Plots the configuration matrix of decay modes (truth vs. predicted)
plt_conf_dm(conf_dm_mat_old, old = True) # Plots the confusion matrix to compare predicted decay modes to old reconstruction method

accuracy = accuracy_calc(conf_dm_mat) # Accuracy calculation of main decay modes (truth vs. predicted)
accuracy = accuracy_calc(conf_dm_mat_old, old = True)
