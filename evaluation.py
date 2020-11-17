import numpy as np

from training import make_generator

def decay_mode_histo(x1, x2, n_tau, dm_bins):
    decay_mode = np.zeros((n_tau,2))
    decay_mode[:,0] = x1
    decay_mode[:,1] = x2
    h_dm, _ = np.histogramdd(decay_mode, bins=[dm_bins,dm_bins])
    h_dm[:,-1] = h_dm[:,4]+h_dm[:,-1] # sum the last and 4. column into the last column
    h_dm = np.delete(h_dm,4,1)        # delete the 4. column
    h_dm[-1,:] = h_dm[4,:]+h_dm[-1,:] # sum the last and 4. column into the last column
    h_dm = np.delete(h_dm,4,0)        # delete the 4. column
    return h_dm

def evaluation(model,generator_xyz,n_steps, n_tau):
    conf_dm_mat = None
    conf_dm_mat_old = None
    ch_bins = np.arange(-0.5,6.5,1)
    dm_bins = [-0.5,0.5,1.5,2.5,3.5,9.5,10.5,11.5,12.5,23.5]
    count_steps = 0

    for x,y,z in generator_xyz():
        y_pred = model.predict(x)
        yy = np.concatenate((y, y_pred), axis = 1)
        # print(y,'\n',y_pred)
        # y is a (n_tau,n_counts) array
        # yy = (y_true_charged, y_true_neutral, y_pred_charged, y_pred_neutral)

        ### Round charge and neutral cosunt to integers:
        y = np.round(y,0)
        y_pred = np.round(y_pred)

        ### Decay mode comparison to new reconstruction:
        h_dm = decay_mode_histo((y[:,0]-1)*5 + y[:,1], (y_pred[:,0]-1)*5 + y_pred[:,1], n_tau, dm_bins)

        ### Decay mode comparison to old reconstruction:
        h_dm_old = decay_mode_histo(z, (y_pred[:,0]-1)*5 + y_pred[:,1], n_tau, dm_bins)

        if conf_dm_mat is None:
            conf_dm_mat = h_dm
            conf_dm_mat_old = h_dm_old
        else:
            conf_dm_mat += h_dm
            conf_dm_mat_old += h_dm_old
            
        count_steps += 1

        if count_steps >= n_steps: break
    
    return conf_dm_mat, conf_dm_mat_old