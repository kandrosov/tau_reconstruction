import tensorflow as tf
import numpy as np
from tensorflow import keras

### All the parameters:
n_tau    = 50  # number of taus (or events) per batch
n_pf     = 100 # number of pf candidates per event
n_fe     = 6   # total muber of features: 6
n_counts = 2   # number of labels per event
n_epoch  = 10  # number of epochs on which to train
n_steps  = 20  # number of steps in the evaluation: (events in conf_dm_mat) = n_steps * n_tau
entry_start = 0
entry_stop  = 10000 # total number of events in the dataset = 14215297
entry_start_val = entry_stop+1
entry_stop_val  = entry_stop+1001


class MyModel(tf.keras.Model):
        def __init__(self):
            super(MyModel, self).__init__()
            self.flatten = tf.keras.layers.Flatten() # flattens a ND array to a 2D array
            self.input_layer = tf.keras.layers.Dense(10, activation=tf.nn.relu) 
            # self.layer1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
            self.output_layer = tf.keras.layers.Dense(2) 
        @tf.function
        def call(self, x):
            x = self.flatten(x)
            x = self.input_layer(x)
            # x = self.layer1(x)
            return self.output_layer(x)


#############################################################################################
##### Custom metrics:
class MyAccuracy(tf.keras.metrics.Metric):

    def __init__(self, name='myaccuracy', **kwargs):
        super(MyAccuracy, self).__init__(name=name, **kwargs) 
        self.accuracy_value = self.add_weight(name="myacc", initializer="zeros")  

    def update_state(self, y_true, y_pred, sample_weight=None):
        dm_bins = [-0.5,0.5,1.5,2.5,3.5,9.5,10.5,11.5,12.5,23.5]

        ### Round to integer:
        y_pred = tf.cast(y_pred, "int32")
        y_true = y_true.numpy()
        # print(y_true)
        y_pred = y_pred.numpy()

        ### Decay mode comparison to new reconstruction:
        h_dm = decay_mode_histo((y_true[:,0]-1)*5 + y_true[:,1], (y_pred[:,0]-1)*5 + y_pred[:,1], dm_bins)

        conf_dm_mat = h_dm

        conf_dm_mat_norm = conf_dm_mat
        for i in range(0,len(conf_dm_mat[0,:])):
            summy = 0
            summy = conf_dm_mat[i,:].sum() # sum of line => normalize lines
            if (summy != 0):
                conf_dm_mat_norm[i,:] = conf_dm_mat[i,:]/summy

        ## Accuracy extraction for important decay modes:
        accuracy = np.zeros((8))
        weights = np.array([0.1151, 0.2593, 0.1081, 0.0118, 0.0980, 0.0476, 0.0051, 0.0029])
        weights = weights/weights.sum()
        self.accuracy_value = 0
        for i in range(0,8):
            accuracy[i] = conf_dm_mat_norm[i,i]
            self.accuracy_value += weights[i]*accuracy[i]
        
    def result(self):    
        return self.accuracy_value


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
        return mse1 + mse2