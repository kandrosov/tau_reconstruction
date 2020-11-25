import numpy as np
import ROOT as R
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pp
import seaborn as sns

from mymodel import *

def evaluation():
    print('\nStart evaluation, load model and generator:\n')
    ##### Reconstruct the model:
    ### Load an empty model:
    data_loader = R.DataLoader('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root', n_tau, entry_start, entry_stop)
    map_features = data_loader.MapCreation()
    model = MyModel(map_features) 
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=CustomMSE(), metrics=[my_acc])#,run_eagerly=True)

    ### Load the last saved model:
    with open('../Models/checkpoint','r') as f:
        line = f.readline()
        mymodelnumber = line.split()  
    f.close()
    a = mymodelnumber[1]
    mymodelnumber = "../Models/"+a[1:-1]
    ### Load the saved weights to the model:
    model.load_weights(mymodelnumber) 

    ### Generator creation:
    generator_xyz, n_batches = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start_test, entry_stop_test, z = True)

    conf_dm_mat = None
    conf_dm_mat_old = None
    dm_bins = [-0.5,0.5,1.5,2.5,3.5,9.5,10.5,11.5,12.5,23.5]
    count_steps = 0

    print('\nStart generator loop to predict:\n')

    for x,y,z in generator_xyz(): # y is a (n_tau,n_labels) array
        y_pred = model.predict(x) 
        yy     = np.concatenate((y, y_pred), axis = 1) # yy = (y_true_charged, y_true_neutral, y_pred_charged, y_pred_neutral)

        ### Round charge and neutral cosunt to integers:
        y      = np.round(y,0)
        y_pred = np.round(y_pred,0)

        ### Decay mode comparison to new reconstruction:
        h_dm = decay_mode_histo((y[:,0]-1)*5 + y[:,1], (y_pred[:,0]-1)*5 + y_pred[:,1], dm_bins)

        ### Decay mode comparison to old reconstruction:
        h_dm_old = decay_mode_histo((y[:,0]-1)*5 + y[:,1], z[:,0], dm_bins)

        if conf_dm_mat is None:
            conf_dm_mat = h_dm
            conf_dm_mat_old = h_dm_old
        else:
            conf_dm_mat += h_dm
            conf_dm_mat_old += h_dm_old
            
        count_steps += 1
        if count_steps % 5000 == 0: print(count_steps)
        if count_steps >= n_steps_test: break
    
    return conf_dm_mat, conf_dm_mat_old


### Accuracy calculation:
def accuracy_calc(conf_dm_mat, old = False):
    ## Normalization of cond_dm_mat:
    conf_dm_mat_norm = conf_dm_mat
    for i in range(0,len(conf_dm_mat[0,:])):
        summy = 0
        summy = conf_dm_mat[i,:].sum() # sum of line => normalize lines
        if (summy != 0):
            conf_dm_mat_norm[i,:] = conf_dm_mat[i,:]/summy

    x_axis_labels = ['$\pi^{\pm}$','$\pi^{\pm} + \pi^0$', '$\pi^{\pm} + 2\pi^0$', \
                 '$\pi^{\pm} + 3\pi^0$','$3\pi^{\pm}$', '$3\pi^{\pm} + 1\pi^0$',\
                 '$3\pi^{\pm} + 2\pi^0$', 'others'] # labels for x-axis
    y_axis_labels = x_axis_labels # labels for y-axis

    fig = plt.figure(figsize=(8,5))
    plt.title("Decay modes")
    sns.heatmap(conf_dm_mat_norm, xticklabels=x_axis_labels, yticklabels=y_axis_labels, cmap='YlGnBu', annot=True, fmt="3.2f")
    plt.ylim(-0.5,9.5)
    plt.xlim(-0.5,9.5)
    if (old == False):
        plt.ylabel('True',fontsize = 16)
    else:
        plt.ylabel('Default tau reconstruction',fontsize = 16)
    plt.xlabel('Predicted',fontsize = 16)
    plt.tight_layout()
    # plt.show()
    if(old==False):
        pdf = pp.PdfPages("../Plots/conf_dm_mat_norm.pdf")
    else:
        pdf = pp.PdfPages("../Plots/conf_dm_mat_norm_old.pdf")
    pdf.savefig(fig)
    pdf.close()
    plt.close()

    ## Accuracy extraction for important decay modes:
    accuracy = np.zeros(8)
    # weights = np.array([0.1151, 0.2593, 0.1081, 0.0118, 0.0980, 0.0476, 0.0051, 0.0029])
    # weights = weights/weights.sum()
    accuracy_value = 0
    for i in range(0,8):
        accuracy[i] = conf_dm_mat_norm[i,i]
        # accuracy_value += weights[i]*accuracy[i]
        accuracy_value += accuracy[i]

    return accuracy_value