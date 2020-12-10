import tensorflow as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output_filename", help="filename where to save the output")
parser.add_argument("--mode", help="mode can be dm or p4 the default is dm")
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True) # dynamic memory allocation

if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]) # ca. 50% => uses effectively 7921MB
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

##################################################################################
##################################################################################
##################################################################################
import ROOT as R

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
_filename = args.output_filename#"/data/cedrine/ModelTest2/test_" # + "mymodel_{}".format(epoch+1) or + "log0.cvs"
parameters = {
      "n_gnn_layers": 10,
      "n_dim_gnn": 2,
      "n_output_gnn": 100,
      "n_output_gnn_last": 10,
      "n_dense_layers": 4,
      "n_dense_nodes": 100,
      "wiring_mode": "m2",
}

history = training(mode = _mode, filename = _filename, parameters=parameters) # trains the model
# history = training(mode = "p4")

plot_metrics(history, mode = _mode) # Plots the loss and accuracy curves for trainset and validationset
# plot_metrics(history, mode = "p4") 

print('#######################################################\
      \n              Training finished !!!                  \n\
#######################################################')