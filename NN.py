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
# print('Accuracy compared to true decay modes: \n',accuracy,'\n')
accuracy = accuracy_calc(conf_dm_mat_old, old = True)

### Check the standartization layers are doing what they should:
# generator_xyz, n_batches = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start, entry_stop, z = True)
# count_steps = 0
# x_a = None
# y_a = None
# for x,y,z in generator_xyz(): # y is a (n_tau,n_counts) array
#     y_pred = model.predict(x) 
#     if x_a is None:
#         x_a = x
#         y_a = y_pred
#     else:
#         x_a = np.append(x_a,x)
#         y_a = np.append(y_a,y_pred)
#     count_steps += 1
#     if count_steps >= n_steps: break
# xx_a = None
# yy_a = None
# for i in range(0,n_tau):
#     if xx_a is None:
#         xx_a = x_a[i,:,3]
#         yy_a = y_a[i,:,3]
#     else:
#         xx_a = np.append(xx_a,x_a[i,:,3])
#         yy_a = np.append(yy_a,y_a[i,:,3])
# print(xx_a)
# print(yy_a)
# print(np.mean(xx_a))
# print(np.std(xx_a))
# print(np.mean(yy_a))
# print(np.std(yy_a))
# # print(max(xx_a))
# # print(min(xx_a))
# # print(max(yy_a))
# # print(min(yy_a))
# plt.hist(xx_a, bins = 100, range=(-2, 2))
# plt.show()
# plt.hist(yy_a, bins = 100, range=(-2, 2))
# plt.show()
