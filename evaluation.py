import numpy as np

from mymodel import *
from training import make_generator

def evaluation(model):
    ### Generator creation:
    generator_xyz, n_batches = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start, entry_stop, z = True)

    conf_dm_mat = None
    conf_dm_mat_old = None
    dm_bins = [-0.5,0.5,1.5,2.5,3.5,9.5,10.5,11.5,12.5,23.5]
    count_steps = 0

    for x,y,z in generator_xyz(): # y is a (n_tau,n_counts) array
        y_pred = model.predict(x) 
        yy     = np.concatenate((y, y_pred), axis = 1) # yy = (y_true_charged, y_true_neutral, y_pred_charged, y_pred_neutral)

        ### Round charge and neutral cosunt to integers:
        y      = np.round(y,0)
        y_pred = np.round(y_pred,0)

        ### Decay mode comparison to new reconstruction:
        h_dm = decay_mode_histo((y[:,0]-1)*5 + y[:,1], (y_pred[:,0]-1)*5 + y_pred[:,1], dm_bins)

        ### Decay mode comparison to old reconstruction:
        h_dm_old = decay_mode_histo((y[:,0]-1)*5 + y[:,1], z, dm_bins)

        if conf_dm_mat is None:
            conf_dm_mat = h_dm
            conf_dm_mat_old = h_dm_old
        else:
            conf_dm_mat += h_dm
            conf_dm_mat_old += h_dm_old
            
        count_steps += 1

        if count_steps >= n_steps: break
    
    return conf_dm_mat, conf_dm_mat_old


### Accuracy calculation:
def accuracy_calc(conf_dm_mat):
    ## Normalization of cond_dm_mat:
    conf_dm_mat_norm = conf_dm_mat
    for i in range(0,len(conf_dm_mat[0,:])):
        summy = 0
        summy = conf_dm_mat[i,:].sum() # sum of line => normalize lines
        if (summy != 0):
            conf_dm_mat_norm[i,:] = conf_dm_mat[i,:]/summy

    # fig = plt.figure(figsize=(5,5))
    # sns.heatmap(conf_dm_mat_norm, cmap='YlGnBu', annot=True, fmt="3.0f")
    # plt.ylim(-0.5,23.5)
    # plt.xlim(-0.5,23.5)
    # plt.ylabel('True decay mode',fontsize = 16)
    # plt.xlabel('Predicted decay mode',fontsize = 16)
    # plt.show()
    # plt.close()
    # pdf = pp.PdfPages("./Plots/conf_dm_mat_norm.pdf")
    # pdf.savefig(fig)
    # pdf.close()

    ## Accuracy extraction for important decay modes:
    accuracy = np.zeros(7)
    weights = np.array([0.1151, 0.2593, 0.1081, 0.0118, 0.0980, 0.0476, 0.0051, 0.0029])
    weights = weights/weights.sum()
    accuracy_value = 0
    for i in range(0,7):
        accuracy[i] = conf_dm_mat_norm[i,i]
        accuracy_value += weights[i]*accuracy[i]

    return accuracy_value