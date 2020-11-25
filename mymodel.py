import ROOT as R
import tensorflow as tf
import numpy as np
from tensorflow import keras
import json
from ROOT import TLorentzVector
import math
# import matplotlib.pyplot as plt

### All the parameters:
n_tau    = 6 #100 # number of taus (or events) per batch
n_pf     = 100  # number of pf candidates per event
n_fe     = 29   # total muber of features: 24
n_labels = 6    # number of labels per event
n_epoch  = 5 #100  # number of epochs on which to train
n_steps_val   = 100#14213
n_steps_test  = 100#63970  # number of steps in the evaluation: (events in conf_dm_mat) = n_steps * n_tau
entry_start   = 0
entry_stop    = 1000#6396973 # total number of events in the dataset = 14'215'297
# 45% = 6'396'973
# 10% = 1'421'351 (approximative calculations have been rounded)
entry_start_val  = entry_stop +1
print('Entry_start_val:', entry_start_val)
entry_stop_val   = entry_stop + n_tau*n_steps_val + 1
print('Entry_stop_val: ',entry_stop_val)
entry_start_test = entry_stop_val+1
entry_stop_test  = entry_stop_val + n_tau*n_steps_test + 1
print('Entry_stop_test (<= 14215297): ',entry_stop_test)



class StdLayer(tf.keras.layers.Layer):
    def __init__(self, file_name, var_pos, n_sigmas, **kwargs):
        with open(file_name) as json_file:
            data_json = json.load(json_file)
        n_vars = len(var_pos)
        self.vars_std = [1] * n_vars
        self.vars_mean = [1] * n_vars
        self.vars_apply = [False] * n_vars
        for var, ms in data_json.items(): 
            pos = var_pos[var]
            self.vars_mean[pos]  = ms[0]['mean']
            self.vars_std[pos]   = ms[0]['std']
            self.vars_apply[pos] = True
        self.vars_mean  = tf.constant(self.vars_mean,  dtype=tf.float32)
        self.vars_std   = tf.constant(self.vars_std,   dtype=tf.float32)
        self.vars_apply = tf.constant(self.vars_apply, dtype=tf.bool)
        self.n_sigmas   = n_sigmas

        super(StdLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StdLayer, self).build(input_shape)

    def call(self, X):
        Y = tf.clip_by_value(( X - self.vars_mean ) / self.vars_std, -self.n_sigmas, self.n_sigmas)
        return tf.where(self.vars_apply, Y, X)

class ScaleLayer(tf.keras.layers.Layer):
    def __init__(self, file_name, var_pos, interval_to_scale, **kwargs):
        with open(file_name) as json_file:
            data_json = json.load(json_file)
        self.a = interval_to_scale[0]
        self.b = interval_to_scale[1]
        n_vars = len(var_pos)
        self.vars_max   = [1] * n_vars
        self.vars_min   = [1] * n_vars
        self.vars_apply = [False] * n_vars
        for var, mm in data_json.items():
            pos = var_pos[var]
            self.vars_min[pos]   = mm[0]['min']
            self.vars_max[pos]   = mm[0]['max']
            self.vars_apply[pos] = True
        self.vars_min   = tf.constant(self.vars_min,   dtype=tf.float32)
        self.vars_max   = tf.constant(self.vars_max,   dtype=tf.float32)
        self.vars_apply = tf.constant(self.vars_apply, dtype=tf.bool)
        self.y = (self.b - self.a) / (self.vars_max - self.vars_min)

        super(ScaleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ScaleLayer, self).build(input_shape)

    def call(self, X):
        Y = tf.clip_by_value( (self.y * ( X - self.vars_min))  + self.a , self.a, self.b)
        return tf.where(self.vars_apply, Y, X)


class MyModel(tf.keras.Model):
    def __init__(self, map_features):
        super(MyModel, self).__init__()
        self.px_index    = map_features["pfCand_px"] 
        self.py_index    = map_features["pfCand_py"] 
        self.pz_index    = map_features["pfCand_pz"]
        self.E_index     = map_features["pfCand_E"] 
        self.valid_index = map_features["pfCand_valid"]

        self.normalize    = StdLayer('mean_std.txt', map_features, 5, name='std_layer')
        self.scale        = ScaleLayer('min_max.txt', map_features, [-1,1], name='scale_layer')
        self.flatten      = tf.keras.layers.Flatten() # flattens a ND array to a 2D array

        self.n_layers = 5 #10
        self.dense = []
        self.dropout_dense = []
        self.batch_norm_dense = []
        self.acti_dense = []
        
        for i in range(0,self.n_layers):
            self.dense.append(tf.keras.layers.Dense(3600, name='dense_{}'.format(i)))
            self.batch_norm_dense.append(tf.keras.layers.BatchNormalization(name='batch_normalization_{}'.format(i)))
            self.acti_dense.append(tf.keras.layers.Activation('relu', name='acti_dense_{}'.format(i)))
            self.dropout_dense.append(tf.keras.layers.Dropout(0.25,name='dropout_dense_{}'.format(i)))
        
        self.output_layer_2   = tf.keras.layers.Dense(2  , name='output_layer_2') 
        self.output_layer_100 = tf.keras.layers.Dense(100, name='output_layer_100', activation='softmax') 

    @tf.function
    def call(self, xx):
        x = self.normalize(xx)
        x = self.scale(x)
        x = self.flatten(x)
        for i in range(0,self.n_layers):
            x = self.dense[i](x)
            x = self.batch_norm_dense[i](x)
            x = self.acti_dense[i](x)
            x = self.dropout_dense[i](x)
        x2   = self.output_layer_2(x)
        x100 = self.output_layer_100(x)

        ### 4-momentum:
        mypxs  = xx[:,:,self.px_index] * x100 * xx[:,:,self.valid_index]
        mypys  = xx[:,:,self.py_index] * x100 * xx[:,:,self.valid_index]
        mypzs  = xx[:,:,self.pz_index] * x100 * xx[:,:,self.valid_index]
        myEs   = xx[:,:,self.E_index]  * x100 * xx[:,:,self.valid_index]

        mypx   = tf.keras.backend.sum(mypxs, axis = 1)
        mypy   = tf.keras.backend.sum(mypys, axis = 1)
        mypz   = tf.keras.backend.sum(mypzs, axis = 1)
        myE    = tf.keras.backend.sum(myEs , axis = 1)

        mypx2  = tf.square(mypx)
        mypy2  = tf.square(mypy)
        mypz2  = tf.square(mypz)

        mypt   = tf.sqrt(mypx2 + mypy2)
        mymass = tf.square(myE) - mypx2 - mypy2 - mypz2
        absp   = tf.sqrt(mypx2 + mypy2 + mypz2)

        ### for myeta and myphi:
        myphi = mypt*0.0
        myeta = mypt*0.0

        cosTheta = tf.where(absp==0, 1.0, mypz/absp)
        myeta = tf.where(cosTheta*cosTheta < 1, -0.5*tf.math.log((1.0-cosTheta)/(1.0+cosTheta)), 0.0)
        myphi = tf.where(tf.math.logical_and(mypx == 0, mypy == 0), 0.0, tf.math.atan2(mypy, mypx))

        xx20 = x2[:,0] # is needed doesn't like direct input
        xx21 = x2[:,1]
        # tf.print('\n\nMass = tf.sqrt(tf.square(myE) - mypx2 - mypy2 - mypz2): ',mymass)
        # tf.print('\ntf.square(myE): ',tf.square(myE))
        # tf.print('\n-mypx2-mypy2-mypz2',-mypx2-mypy2-mypz2)
        # print('Shapes: ', xx20.shape, ' ', xx21.shape, ' ', mypt.shape, ' ', myeta.shape, ' ', myphi.shape, ' ', mymass.shape, ' ',)
        xout = tf.stack([xx20,xx21,mypt,myeta,myphi,mymass], axis=1)
        # print(xout)
        return xout

 
### Function that creates generators:
def make_generator(file_name, entry_begin, entry_end, z = False):
    _data_loader = R.DataLoader(file_name, n_tau, entry_begin, entry_end)
    _n_batches   = _data_loader.NumberOfBatches()

    def _generator():
        cnt = 0
        while True:
            _data_loader.Reset()
            while _data_loader.HasNext(): 
                data = _data_loader.LoadNext()
                x_np = np.asarray(data.x)
                x_3d = x_np.reshape((n_tau, n_pf, n_fe))
                y_np = np.asarray(data.y)
                y_2d = y_np.reshape((n_tau, n_labels))
                if z == True:
                    z_np = np.asarray(data.z)
                    z_2d = z_np.reshape((n_tau, n_labels-1))
                    yield x_3d, y_2d, z_2d
                else:
                    yield x_3d, y_2d
            ++cnt
            if cnt == 100: 
                gc.collect() # garbage collection to improve preformance
                cnt = 0
    
    return _generator, _n_batches

#############################################################################################
##### Custom metrics:
### Accuracy calculation for number of charged/neutral hadrons:
@tf.function
def my_acc(y_true, y_pred):
    y_true = tf.math.round(y_true)
    y_true_int = tf.cast(y_true, tf.int32)
    y_pred = tf.math.round(y_pred)
    y_pred_int = tf.cast(y_pred, tf.int32)
    result = tf.math.logical_and(y_true_int[:, 0] == y_pred_int[:, 0], y_true_int[:, 1] == y_pred_int[:, 1])
    return tf.cast(result, tf.float32)

### Resolution of 4-momentum:
def my_resolution(y_true, y_pred):
    pt_res   = (y_pred[:,2]-y_true[:,2])/y_true[:,2]
    eta_res  = (y_pred[:,3]-y_true[:,3])
    phi_res  = (y_pred[:,4]-y_true[:,4])
    mass_res = (y_pred[:,5]-y_true[:,5])
    pt_std   = tf.keras.backend.std(pt_res)
    eta_std  = tf.keras.backend.std(eta_res)
    phi_std  = tf.keras.backend.std(phi_res)
    mass_std = tf.keras.backend.std(mass_res)
    return pt_std,eta_std,phi_std, mass_std


def decay_mode_histo(x1, x2, dm_bins):
    decay_mode = np.zeros((x1.shape[0],2))
    decay_mode[:,0] = x1
    decay_mode[:,1] = x2
    h_dm, _ = np.histogramdd(decay_mode, bins=[dm_bins,dm_bins])
    h_dm[:,-1] = h_dm[:,4]+h_dm[:,-1] # sum the last and 4. column into the last column
    h_dm = np.delete(h_dm,4,1)        # delete the 4. column
    h_dm[-1,:] = h_dm[4,:]+h_dm[-1,:] # sum the last and 4. column into the last column
    h_dm = np.delete(h_dm,4,0)        # delete the 4. column
    return h_dm


##### Custom loss function:
class CustomMSE(keras.losses.Loss):
    def __init__(self, name="custom_mse"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        mse1 = tf.math.reduce_mean(tf.square(y_true[:,0] - y_pred[:,0]))
        mse2 = tf.math.reduce_mean(tf.square(y_true[:,1] - y_pred[:,1]))
        mse3 = tf.math.reduce_mean(tf.square(y_true[:,2] - y_pred[:,2]))
        mse4 = tf.math.reduce_mean(tf.square(y_true[:,3] - y_pred[:,3]))
        mse5 = tf.math.reduce_mean(tf.square(y_true[:,4] - y_pred[:,4]))
        mse6 = tf.math.reduce_mean(tf.square(y_true[:,5] - y_pred[:,5]))
        return mse1 + mse2 + mse3 + mse4 + mse5 + mse6

class ValidationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # keys = list(logs.keys())
        # print("Log keys: {}".format(keys))
        self.myresults = np.zeros(2)
        generator_val, n_batches_val = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start_val, entry_stop_val) 
        self.myresults = self.model.evaluate(x = tf.data.Dataset.from_generator(generator_val,(tf.float32, tf.float32),\
                            (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,n_labels]))), batch_size = n_batches_val, steps = n_steps_val)
        cnt = 0
        for i in self.model.metrics_names:
            logs["val_"+i] = self.myresults[cnt]
            cnt = cnt + 1
   
        

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor   = 'val_loss', 
        min_delta = 0, # Minimum change in the monitored quantity to qualify as an improvement
        patience  = 5, # Number of epochs with no improvement after which training will be stopped
        verbose   = 0, 
        mode      = 'min', # it will stop when the quantity monitored has stopped decreasing
        baseline  = None, 
        restore_best_weights = False,
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath       = "../Models/mymodel_{epoch}",
        save_freq      = 'epoch',
        verbose        = 1, 
        save_weights_only = True,
    ),
    tf.keras.callbacks.CSVLogger(
        filename  = '../CSV_run_logs/log.csv', 
        separator = ',', 
        append    = False,
    )
]
