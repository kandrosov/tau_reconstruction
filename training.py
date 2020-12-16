import ROOT as R
import numpy as np
import tensorflow as tf
import os

from mymodel import *


### This function compiles and trains the model:
def training(mode, filename, parameters):


    print('\nLoading of model and generators:\n')
    _map_features = R.DataLoader.MapCreation()
    mapOfy = R.DataLoader.MapCreationy()

    # print('map of features in training: ',map_features)
    model = MyGNN(mode=mode, map_features = _map_features, filename = filename, **parameters) # creates the model

    ### Generator creation:
    generator, n_batches = make_generator(entry_start, entry_stop)

    print('\nCompilation of model:\n')
    CustomMSE.mode = mode
    metrics = []
    if "dm" in mode:
        # metrics.extend([my_cat_dm])
        metrics.extend([my_acc, my_mse_ch, my_mse_neu])
        # metrics.extend([my_acc, my_mse_ch_4, my_mse_neu_4])
        # metrics.extend([my_acc, my_mse_ch_new, my_mse_neu_new])
    if "p4" in mode:
        metrics.extend([my_mse_pt, my_mse_mass, pt_res, pt_res_rel, m2_res])
        # metrics.extend([my_mse_pt, my_mse_mass, pt_res, pt_res_rel, m2_res, log_cosh_pt, log_cosh_mass])
        # metrics.extend([my_mse_pt, my_mse_mass, pt_res, pt_res_rel, m2_res, log_cosh_mass])
        # metrics.extend([my_mse_pt_4, my_mse_mass_4, pt_res, pt_res_rel, m2_res])
        # metrics.extend([my_mse_pt, my_mse_mass, pt_res, pt_res_rel, m2_res, quantile_pt])
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=CustomMSE(), metrics=metrics, run_eagerly=True)

    # model.build((n_tau, n_pf, n_fe))
    # model.summary()
    print('\nModel is fit:\n')

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(filename,"log0.csv"), append=False, separator=',')

    history = model.fit(x = tf.data.Dataset.from_generator(generator,(tf.float32, tf.float32),
                        (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,4]))),
                        epochs = n_epoch, verbose=1, steps_per_epoch = n_batches, callbacks=[ValidationCallback(),early_stop, csv_logger])

    print('\nModel finished training.')

    print('\nFinished model running.\n')

    return history
