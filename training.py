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
    # generator_val, n_batches_val = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start_val, entry_stop_val) 
    
    # print('\nSave validation set in memory:\n') #takes a long time
    # ### Save validation set in memory:
    # x_val = None
    # y_val = None
    # count_steps = 0
    # for x,y in generator_val():
    #     if x_val is None:
    #         x_val = x
    #         y_val = y
    #     else:
    #         x_val = np.append(x_val,x, axis = 0)
    #         y_val = np.append(y_val,y, axis = 0)
    #     count_steps += 1
    #     if count_steps % 1000 == 0: print(count_steps)
    #     if count_steps >= n_steps_val: break
    
    print('\nCompilation fo model:\n')
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=CustomMSE(), metrics=[MyAccuracy()], run_eagerly=True)
    # run_eagerly=True is there to convert tensors to numpy in mymodel.py

    print('\nModel is fit:\n')
    history = model.fit(x = tf.data.Dataset.from_generator(generator,(tf.float32, tf.float32),\
                            (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,n_counts]))),\
                            epochs = n_epoch, steps_per_epoch = n_batches, callbacks=[ValidationCallback()])
    
    model.build((n_tau, n_pf, n_fe)) #(50,2400)
    model.summary()

    model.save("../Models/my_model") # works

    print('\nFinished model running:\n')

    return history