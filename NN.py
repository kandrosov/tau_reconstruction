import ROOT as R
import numpy as np
import tensorflow as tf

R.gInterpreter.ProcessLine('#include "DataLoader.h"')

n_tau = 10
n_pf  = 5
n_fe  = 30
n_counts = 2

data_loader = R.DataLoader('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root', n_tau) 

data = data_loader.LoadNext()
n_batches = data_loader.NumberOfBatches()

def generator():
    while True:
        data_loader.Reset()
        while data_loader.HasNext():
            data = data_loader.LoadNext()
            x_np = np.asarray(data.x)
            x_3d = x_np.reshape((n_tau, n_pf, n_fe))
            y_np = np.asarray(data.y)
            y_2d = y_np.reshape((n_tau, n_counts))
            yield x_3d, y_2d

# Model construction:
inputs = tf.keras.Input(shape=(n_pf,n_fe))
x_inputs = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(2, activation=tf.nn.relu)(x_inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

model.fit(x = tf.data.Dataset.from_generator(generator,\
                                            (tf.float32, tf.float32),\
                                            (tf.TensorShape([None,5,30]), tf.TensorShape([None,2]))), \
                                            epochs = 1, steps_per_epoch = n_batches)
