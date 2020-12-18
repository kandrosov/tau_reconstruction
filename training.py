import tensorflow as tf

from mymodel import *


### This function compiles and trains the model:
def training(mode, filename, parameters):
    # epoch_resume = parameters['epoch_resume']
    # n_stop = parameters['n_stop']
    # print(type(epoch_resume))
    # print(type(n_stop))

    print('\nLoading of model and generators:\n')

    ### Generator creation:
    generator, n_batches = make_generator(entry_start, entry_stop)

    _map_features = R.DataLoader.MapCreation()
    model = MyGNN(mode=mode, map_features = _map_features, filename = filename, **parameters) # creates the model
    ### Metrics:
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
    print('\nCompilation of model:\n')
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=CustomMSE(), metrics=metrics, run_eagerly=True)
    
    model.build((n_tau, n_pf, n_fe))
    model.summary()
    print('\nModel created\n')

    # if(epoch_resume < 0):
    #     #### Normal training:
    #     _map_features = R.DataLoader.MapCreation()
    #     model = MyGNN(mode=mode, map_features = _map_features, filename = filename, **parameters) # creates the model
    #     ### Metrics:
    #     CustomMSE.mode = mode
    #     metrics = []
    #     if "dm" in mode:
    #         # metrics.extend([my_cat_dm])
    #         metrics.extend([my_acc, my_mse_ch, my_mse_neu])
    #         # metrics.extend([my_acc, my_mse_ch_4, my_mse_neu_4])
    #         # metrics.extend([my_acc, my_mse_ch_new, my_mse_neu_new])
    #     if "p4" in mode:
    #         metrics.extend([my_mse_pt, my_mse_mass, pt_res, pt_res_rel, m2_res])
    #         # metrics.extend([my_mse_pt, my_mse_mass, pt_res, pt_res_rel, m2_res, log_cosh_pt, log_cosh_mass])
    #         # metrics.extend([my_mse_pt, my_mse_mass, pt_res, pt_res_rel, m2_res, log_cosh_mass])
    #         # metrics.extend([my_mse_pt_4, my_mse_mass_4, pt_res, pt_res_rel, m2_res])
    #         # metrics.extend([my_mse_pt, my_mse_mass, pt_res, pt_res_rel, m2_res, quantile_pt])
    #     print('\nCompilation of model:\n')
    #     model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=CustomMSE(), metrics=metrics, run_eagerly=True)
    #     model.build((n_tau, n_pf, n_fe))
    #     model.summary()
    #     print('\nModel created\n')
    #     epoch_number = n_stop
    #     epoch_resume = 0
    # else:
    #     #### Load model:
    #     CustomMSE.mode = mode
    #     custom_objects = {
    #         "CustomMSE": CustomMSE,
    #         # "my_cat_dm" : my_cat_dm,
    #         "my_acc" : my_acc,
    #         "my_mse_ch"   : my_mse_ch,
    #         "my_mse_neu"  : my_mse_neu,
    #         "my_mse_pt"   : my_mse_pt,
    #         "my_mse_mass" : my_mse_mass,
    #         "pt_res" : pt_res,
    #         "m2_res" : m2_res,
    #         "pt_res_rel" : pt_res_rel,
    #         # "my_mse_ch_new"   : my_mse_ch_new,
    #         # "my_mse_neu_new"  : my_mse_neu_new,
    #         # "quantile_pt" : quantile_pt,
    #         # "log_cosh_pt" : log_cosh_pt,
    #         # "log_cosh_mass" : log_cosh_mass,
    #         # "my_mse_ch_4"   : my_mse_ch_4,
    #         # "my_mse_neu_4"  : my_mse_neu_4,
    #         # "my_mse_pt_4"   : my_mse_pt_4,
    #         # "my_mse_mass_4" : my_mse_mass_4,
    #     }
    #     model = tf.keras.models.load_model(os.path.join(filename,"my_model_{}".format(epoch_resume)), custom_objects=custom_objects, compile=True)
    #     print('\nModel loaded\n')
    #     epoch_number = epoch_resume + n_stop

    print('\nModel is fit:\n')
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(filename,"log0.csv"), append=True, separator=',')

    history = model.fit(x = tf.data.Dataset.from_generator(generator,(tf.float32, tf.float32),
                        (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,4]))),
                        epochs = n_epoch, verbose=1, steps_per_epoch = n_batches, callbacks=[ValidationCallback(),early_stop, csv_logger])
    # history = model.fit(x = tf.data.Dataset.from_generator(generator,(tf.float32, tf.float32),
    #                     (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,4]))),
    #                     epochs = epoch_number, verbose=1, steps_per_epoch = n_batches, initial_epoch = epoch_resume-1,
    #                     callbacks=[ValidationCallback(),early_stop, csv_logger])

    print('\nModel finished training.')

    print('\nFinished model running.\n')

    return history
