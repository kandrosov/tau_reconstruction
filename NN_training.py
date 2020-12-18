import tensorflow as tf
import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output_filename", help="filename where to save the output")
parser.add_argument("--mode", help="mode can be dm, p4 or p4_dm the default is dm")
parser.add_argument("--params", help="model parameters. Example: --params { 'n_gnn_layers': 5, 'n_dim_gnn': 2, 'n_output_gnn': 50, 'n_output_gnn_last': 50, 'n_dense_layers': 4, 'n_dense_nodes': 200, 'wiring_mode': 'm3', 'dropout_rate' : 0.2 , 'regu_rate':0.01}. no dropout is applied for dropout_rate = 0, and no regularization is applied for regu_rate < 0.")
args = parser.parse_args()
parameters = json.loads(args.params)

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True) # dynamic memory allocation

if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)]) # ca. 50% => uses effectively 7921MB
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

##################################################################################
##################################################################################
##################################################################################
import ROOT as R
import os
import sys

R.gInterpreter.ProcessLine('#include "DataLoader.h"')

from training import training
from mymodel import * # gives us all n_tau, n_pf, ....
from plotting import plot_metrics

print('\n#######################################################\
      \n              Training start !!!                  \n\
#######################################################')

###### Parameters:
if args.mode:
      _mode = args.mode
else:
      _mode = "dm"
_filename = args.output_filename #"/data/results/run1/" # + "mymodel_{}".format(epoch+1) or + "log0.cvs"

### Create the directory:
try:
    os.mkdir(_filename)
except OSError:
    print ("Creation of the directory %s failed" % _filename)
    sys.exit()
else:
    print ("Successfully created the directory %s " % _filename)

R.DataLoader.Initialize('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50_v2.root')

history = training(mode = _mode, filename = _filename, parameters=parameters) # trains the model

plot_metrics(history, mode = _mode, filename = _filename) # Plots the loss and accuracy curves for trainset and validationset

print('#######################################################\
      \n              Training finished !!!                  \n\
#######################################################')
