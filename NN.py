import ROOT as R

R.gInterpreter.ProcessLine('#include "DataLoader.h"')

from training import training, make_generator
from mymodel import * #gives us all n_tau, n_pf, ....
from plotting import plot_metrics, plt_conf_dm
from evaluation import evaluation, accuracy_calc


model = MyModel() # creates the model

history = training(model) # trains the model

plot_metrics(history) # Plots the loss and accuracy curves for trainset and validationset

conf_dm_mat, conf_dm_mat_old = evaluation(model) # creates the confusion matrices

plt_conf_dm(conf_dm_mat)  # Plots the configuration matrix of decay modes (truth vs. predicted)
# plt_conf_dm(conf_dm_mat_old, old = True) # Plots the confusion matrix to compare predicted decay modes to old reconstruction method

accuracy = accuracy_calc(conf_dm_mat) # Accuracy calculation of main decay modes (truth vs. predicted)
print('Accuracy compared to true decay modes: \n',accuracy,'\n')
