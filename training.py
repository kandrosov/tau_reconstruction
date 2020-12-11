import ROOT as R
import numpy as np
import tensorflow as tf
import os

from mymodel import *


### This function compiles and trains the model:
def training(mode, filename, parameters):
        

    print('\nLoading of model and generators:\n')
    data_loader = R.DataLoader('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root', n_tau, entry_start, entry_stop)
    _map_features = data_loader.MapCreation()
    mapOfy = data_loader.MapCreationy()

    # print('map of features in training: ',map_features)
    model = MyGNN(mode=mode, map_features = _map_features, filename = filename, **parameters)#parameters = parameters) # creates the model

    ### Generator creation:
    generator, n_batches = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start, entry_stop)
    
    print('\nCompilation of model:\n')
    CustomMSE.mode = mode
    if(mode=="dm"):
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=CustomMSE(), 
                        metrics=[my_acc, my_mse_ch, my_mse_neu])
    else:
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=CustomMSE(), 
                        metrics=[my_mse_pt, my_mse_mass, pt_res, m2_res])#, run_eagerly=True)


    model.build((n_tau, n_pf, n_fe))
    model.summary()
    print('\nModel is fit:\n')

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(filename,"log0.csv"), append=False, separator=',')

    history = model.fit(x = tf.data.Dataset.from_generator(generator,(tf.float32, tf.float32),
                        (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,4]))),
                        epochs = n_epoch, steps_per_epoch = n_batches, callbacks=[ValidationCallback(),early_stop, csv_logger])
    
    print('\nModel finished training.')

    print('\nFinished model running.\n')

    return history