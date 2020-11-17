import ROOT as R
import numpy as np
import tensorflow as tf

def make_generator(file_name, entry_start, entry_stop, n_tau, n_pf, n_fe, n_counts, z = False):
    data_loader = R.DataLoader(file_name, n_tau, entry_start, entry_stop)
    n_batches = data_loader.NumberOfBatches()

    def generator():
        cnt = 0
        while True:
            data_loader.Reset()
            while data_loader.HasNext():
                data = data_loader.LoadNext()
                x_np = np.asarray(data.x)
                x_3d = x_np.reshape((n_tau, n_pf, n_fe))
                y_np = np.asarray(data.y)
                y_2d = y_np.reshape((n_tau, n_counts))
                if z == True:
                    z_1d = np.asarray(data.z)
                    yield x_3d, y_2d, z_1d
                else:
                    yield x_3d, y_2d
            ++cnt
            if cnt == 1000: 
                gc.collect() # garbage collection to improve preformance
                cnt = 0
    
    return generator, n_batches

def training(model, generator, generator_val, n_tau, n_pf, n_fe, n_counts, n_epoch, n_batches,n_steps):

    x_val = None
    y_val = None
    count_steps = 0
    for x,y in generator_val():
        if x_val is None:
            x_val = x
            y_val = y
            print(x.shape)
            print(y.shape)
        else:
            x_val = np.append(x_val,x, axis = 0)
            y_val = np.append(y_val,y, axis = 0)
        count_steps += 1

        if count_steps >= n_steps: break

    print(x_val.shape)
    print(y_val.shape)

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae","accuracy"])
    history = model.fit(x = tf.data.Dataset.from_generator(generator,(tf.float32, tf.float32),\
                            (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,n_counts]))),\
                            validation_data = (x_val,y_val), epochs = n_epoch, steps_per_epoch = n_batches)

    model.summary()

    return history