import tensorflow as tf
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("saved_filename", help="filename where the model was saved and where the figures will be saved. Example: '/data/results/run1/' ")
parser.add_argument("--mode", help="mode can be 'dm' or 'p4'. The default is 'dm'")
parser.add_argument("--epoch_number", help="the epoch to evaluate. Default is 1.")
args = parser.parse_args()

######## Memory allocation:
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True) # dynamic memory allocation

if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]) # => uses effectively 6367 MB
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

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

_filename = args.saved_filename #"/data/results/run1/" # + "mymodel_{}".format(epoch_number)

if args.mode:
      _mode = args.mode
else:
      _mode = "dm"

if args.epoch_number:
      _epoch_number = args.epoch_number
else:
      _epoch_number = 1

_filename_plots = os.path.join(_filename, "Plots_"+str(_epoch_number))
try:
    os.mkdir(_filename_plots)
except OSError:
    print ("Creation of the directory %s failed, because already existing" % _filename)
else:
    print ("Successfully created the directory %s " % _filename)

# The number over which it is evaluated is decided in mymodel.py: n_steps_test is the number of batches in the test.

CustomMSE.mode = _mode

if(_mode=="dm" or _mode =="p4_dm"):
      conf_dm_mat, conf_dm_mat_old = evaluation(mode = _mode, filename = _filename, epoch_number= _epoch_number) #creates the confusion matrices

      plt_conf_dm(conf_dm_mat, filename_plots = _filename_plots)  # Plots the configuration matrix of decay modes (truth vs. predicted)
      plt_conf_dm(conf_dm_mat_old, filename_plots = _filename_plots, old = True) # Plots the confusion matrix to compare predicted decay modes to old reconstruction method

      print('\nConfusion matrices finished.')

      accuracy = accuracy_calc(conf_dm_mat, filename_plots = _filename_plots) # Accuracy calculation of main decay modes (truth vs. predicted)
      accuracy = accuracy_calc(conf_dm_mat_old, filename_plots = _filename_plots, old = True)

elif(_mode=="p4"):
      evaluation(mode = _mode, filename = _filename, epoch_number= _epoch_number) #creates the resolution plots
else:
      print('Mode not possible!!!')

print('#######################################################\
      \n            Evaluation finished !!!                 \n\
#######################################################')
