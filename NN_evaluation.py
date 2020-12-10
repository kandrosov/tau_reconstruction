import tensorflow as tf

######## Memory allocation:
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True) # dynamic memory allocation

# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9000)]) # => uses effectively 5367 MB
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

import ROOT as R
import matplotlib.pyplot as plt
import numpy as np

R.gInterpreter.ProcessLine('#include "DataLoader.h"')

from mymodel import * 
from plotting import plot_metrics, plt_conf_dm, accuracy_calc
from evaluation import evaluation

print('\n#######################################################\
      \n            Evaluation start !!!                  \n\
#######################################################')

_mode = "dm"
# _mode = "p4"
_filename = "/data/cedrine/ModelTest2/test_" # + "mymodel_{}".format(epoch_number)
_epoch_number = 2



conf_dm_mat, conf_dm_mat_old = evaluation(mode = _mode, filename = _filename, epoch_number= _epoch_number) # creates the confusion matrices

print('\nEvaluation is finished.\n')

plt_conf_dm(conf_dm_mat)  # Plots the configuration matrix of decay modes (truth vs. predicted)
plt_conf_dm(conf_dm_mat_old, old = True) # Plots the confusion matrix to compare predicted decay modes to old reconstruction method

print('\nConfusion matrices finished.')

accuracy = accuracy_calc(conf_dm_mat) # Accuracy calculation of main decay modes (truth vs. predicted)
accuracy = accuracy_calc(conf_dm_mat_old, old = True)

print('#######################################################\
      \n            Plotting finished !!!                  \n\
#######################################################')
