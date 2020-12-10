import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]) # ca. 50% => uses effectively 7921MB
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

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

history = training() # trains the model

plot_metrics(history) # Plots the loss and accuracy curves for trainset and validationset

print('#######################################################\
      \n              Training finished !!!                  \n\
#######################################################')
