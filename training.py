import ROOT as R
import numpy as np
import tensorflow as tf

from mymodel import *


### This function compiles and trains the model:
def training():
    print('\nLoading of model and generators:\n')
    data_loader = R.DataLoader('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root', n_tau, entry_start, entry_stop)
    map_features = data_loader.MapCreation()

    model = MyModel(map_features) # creates the model

    ### Generator creation:
    generator, n_batches         = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start, entry_stop)
    
    print('\nCompilation of model:\n')
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=CustomMSE(), metrics=[my_acc,my_resolution])#,run_eagerly=True)

    print('\nModel is fit:\n')
    history = model.fit(x = tf.data.Dataset.from_generator(generator,(tf.float32, tf.float32),\
                            (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,n_labels]))),\
                            epochs = n_epoch, steps_per_epoch = n_batches, callbacks=[ValidationCallback(),callbacks])
    
    model.build((n_tau, n_pf, n_fe))
    model.summary()

    print('\nFinished model running:\n')

    return history