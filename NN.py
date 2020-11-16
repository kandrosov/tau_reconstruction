import ROOT as R
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf as pp

def main():
    R.gInterpreter.ProcessLine('#include "DataLoader.h"')

    n_tau    = 10
    n_pf     = 100
    n_fe     = 6
    n_counts = 2
    n_epoch  = 5

    data_loader = R.DataLoader('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root', n_tau, 0, 1000) 
    n_batches = data_loader.NumberOfBatches()

    #### Model construction:
    class MyModel(tf.keras.Model):

        def __init__(self):
            super(MyModel, self).__init__()
            self.flatten = tf.keras.layers.Flatten() #flattens a ND array to a 2D array
            self.input_layer = tf.keras.layers.Dense(20, activation=tf.nn.relu) 
            self.layer1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
            self.output_layer = tf.keras.layers.Dense(2) 

        def call(self, x):
            x = self.flatten(x)
            x = self.input_layer(x)
            x = self.layer1(x)
            return self.output_layer(x)

    model = MyModel()

    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae","accuracy"])

    history = model.fit(x = tf.data.Dataset.from_generator(generator,(tf.float32, tf.float32), (tf.TensorShape([None,n_pf,n_fe]), tf.TensorShape([None,n_counts]))), epochs = n_epoch, steps_per_epoch = n_batches)

    model.summary()

    plot_metrics(history, n_epoch) # Plots the loss and accuracy curves


    ##### Physics performance of the model:
    data_loader = R.DataLoader('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root', n_tau, 10000, 11000)
    n_batches   = data_loader.NumberOfBatches()

    conf_matrix = None
    conf_dm_mat = None
    ch_bins = np.arange(-0.5,6.5,1)
    dm_bins = np.arange(-0.5,23.5,1) 
    n_steps = 100
    count_steps = 0

    for x,y in generator():
        y_pred = model.predict(x)
        yy = np.concatenate((y, y_pred), axis = 1)
        h, _ = np.histogramdd(yy, bins=[ch_bins, ch_bins, ch_bins, ch_bins])
        # print(y) #is a (n_tau,n_counts) array
        # print(y[:,0]) # gives the first column
        # print(yy) # yy = (y_true_charged, y_true_neutral, y_pred_charged, y_pred_neutral)

        ### Decay mode
        decay_mode = np.zeros((10,2))
        decay_mode[:,0] = (y[:,0]-1)*5 + y[:,1]
        decay_mode[:,1] = (y_pred[:,0]-1)*5 + y_pred[:,1]
        h_dm, _ = np.histogramdd(decay_mode, bins=[dm_bins,dm_bins])

        if conf_matrix is None:
            conf_matrix = h
            conf_dm_mat = h_dm
        else:
            conf_matrix += h
            conf_dm_mat += h_dm
        
        count_steps += 1

        if count_steps >= n_steps: break

    ### Plot the different physics performances:
    plt_conf_charge(conf_matrix)          # Configuration matrix of charged counts
    plt_conf_neutral(conf_matrix)         # Configuration matrix of neutral counts
    plt_conf_dm(conf_dm_mat)              # Configuration matrix of decay modes
    accuracy = accuracy_calc(conf_dm_mat) # Accuracy calculation of main decay modes

    print('Accuracy for the decay modes [0,1,2,3,10,11,12]: \n',accuracy,'\n')


################################################################################
################################################################################
##### Functions:
################################################################################
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

### Plots the loss function and accuracy for the differenc epochs:
def plot_metrics(history, n_epoch):
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    mse = hist["mae"]
    acc = hist["accuracy"]

    fig0, axes = plt.subplots(2, sharex=False, figsize=(12, 8))
    fig0.suptitle('Training Metrics')
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].set_xlabel("Epoch", fontsize=14)
    axes[0].set_xticks(np.arange(0, n_epoch, 1.0))
    axes[0].plot(epochs, mse, 'bo', label="Loss")
    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].set_xticks(np.arange(0, n_epoch, 1.0))
    axes[1].plot(epochs, acc, 'ro', label="Loss")
    # plt.show()
    pdf0 = pp.PdfPages("./Plots/Metrics.pdf")
    pdf0.savefig(fig0)
    pdf0.close()
    plt.close()

################################################################################
##### Functions to plot physics performance of the model:
### Confusion matrix of charged count:
def plt_conf_charge(conf_matrix):
    conf_mat_1 = np.zeros((6,6))

    for i in range(0,6):
        for j in range(0,6):
            conf_mat_1 += conf_matrix[:,i,:,j]

    fig = plt.figure(figsize=(5,5))
    sns.heatmap(conf_mat_1, cmap='YlGnBu', annot=True, fmt="3.0f")
    plt.ylim(-0.5,6.5)
    plt.xlim(-0.5,6.5)
    plt.ylabel('True charge counts',fontsize = 16)
    plt.xlabel('Predicted charge counts',fontsize = 16)
    # plt.show()
    plt.close()
    pdf = pp.PdfPages("./Plots/conf_mat_charged.pdf")
    pdf.savefig(fig)
    pdf.close()

### Confusion matrix of neutral count:
def plt_conf_neutral(conf_matrix):
    conf_mat_1 = np.zeros((6,6))

    for i in range(0,6):
        for j in range(0,6):
            conf_mat_1 += conf_matrix[i,:,j,:]

    fig = plt.figure(figsize=(5,5))
    sns.heatmap(conf_mat_1, cmap='YlGnBu', annot=True, fmt="3.0f")
    plt.ylim(-0.5,6.5)
    plt.xlim(-0.5,6.5)
    plt.ylabel('True neutral counts',fontsize = 16)
    plt.xlabel('Predicted neutral counts',fontsize = 16)
    # plt.show()
    plt.close()
    pdf = pp.PdfPages("./Plots/conf_mat_neutral.pdf")
    pdf.savefig(fig)
    pdf.close()

### Confusion matrix of decay modes:
def plt_conf_dm(conf_dm_mat):
    fig = plt.figure(figsize=(8,5))
    sns.heatmap(conf_dm_mat, cmap='YlGnBu', annot=True, fmt="3.0f")
    plt.ylim(-0.5,23.5)
    plt.xlim(-0.5,23.5)
    plt.ylabel('True decay mode counts',fontsize = 16)
    plt.xlabel('Predicted decay mode counts',fontsize = 16)
    # plt.show()
    plt.close()
    pdf = pp.PdfPages("./Plots/conf_dm_mat.pdf")
    pdf.savefig(fig)
    pdf.close()

    ### check the true distribution of the decay modes:
    tot_dm = conf_dm_mat.sum(axis=1)
    print('\n Total number of events with a decay mode for true, corresponds to [0,1,2,3,...] decay mode: \n',tot_dm,'\n') 
    tot_dm_norm = tot_dm/tot_dm.sum()
    print('Probabilities of decay mode for true: \n',tot_dm_norm,'\n') 

### Accuracy calculation:
def accuracy_calc(conf_dm_mat):
    ## Normalization of cond_dm_mat:
    conf_dm_mat_norm = conf_dm_mat
    for i in range(0,len(conf_dm_mat[0,:])):
        summy = 0
        summy = conf_dm_mat[i,:].sum() # sum of line => normalize lines
        if (summy != 0):
            conf_dm_mat_norm[i,:] = conf_dm_mat[i,:]/summy

    fig = plt.figure(figsize=(8,5))
    sns.heatmap(conf_dm_mat_norm, cmap='YlGnBu', annot=True, fmt="3.0f")
    plt.ylim(-0.5,23.5)
    plt.xlim(-0.5,23.5)
    plt.ylabel('True decay mode counts',fontsize = 16)
    plt.xlabel('Predicted decay mode counts',fontsize = 16)
    # plt.show()
    plt.close()
    pdf = pp.PdfPages("./Plots/conf_dm_mat_norm.pdf")
    pdf.savefig(fig)
    pdf.close()

    ## Accuracy extraction for important decay modes:
    accuracy = np.zeros(7)
    accuracy[0] = conf_dm_mat_norm[0,0]
    accuracy[1] = conf_dm_mat_norm[1,1]
    accuracy[2] = conf_dm_mat_norm[2,2]
    accuracy[3] = conf_dm_mat_norm[3,3]
    accuracy[4] = conf_dm_mat_norm[10,10]
    accuracy[5] = conf_dm_mat_norm[11,11]
    accuracy[6] = conf_dm_mat_norm[12,12]

    return accuracy

################################################################################
if __name__ == '__main__':
    main()