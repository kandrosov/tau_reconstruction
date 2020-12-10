import numpy as np
import ROOT as R
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pp
import seaborn as sns

from mymodel import *
from plotting import plot_res

def make_sqrt(m):
    if(m>=0): 
        m = math.sqrt(m)
    else: 
        m = -math.sqrt(-m)
    return m

def evaluation(mode, filename, epoch_number):
    print('\nStart evaluation, load model and generator:\n')
    ##### Load the model:
    if(mode== "dm"):
        custom_objects = {
            "CustomMSE": CustomMSE,
            "my_acc" : my_acc,
            "my_mse_ch"   : my_mse_ch,
            "my_mse_neu"  : my_mse_neu,
        }
    elif(mode=="p4"):
        custom_objects = {
            "CustomMSE": CustomMSE,
            "pt_res" : pt_res,
            "m2_res" : m2_res,
            "my_mse_pt"   : my_mse_pt,
            "my_mse_mass" : my_mse_mass,
        }

    model = tf.keras.models.load_model(filename +"my_model_{}".format(epoch_number), custom_objects=custom_objects, compile=True)
    print("Model loaded.")

    generator_xyz, n_batches = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start_test, entry_stop_test, z = True)

    count_steps = 0

    if(mode=="p4"):
        h_pt_tot = R.TH1F('h_pt', 'pt resolution', 200, -2, 1.5)#nbins, xlow, xup
        h_m2_tot = R.TH1F('h_m2', 'm2 resolution', 200, -1, 1)

        def_h_pt_tot = R.TH1F('def_h_pt', 'default pt resolution'  , 200, -2, 1.5)
        def_h_m2_tot = R.TH1F('def_h_m2', 'default m2 resolution', 200, -1, 1)

        print('\nStart generator loop to predict p4 resolutions:\n')
        for x,y,z in generator_xyz(): # y is a (n_tau,n_labels) array

            y_pred = model.predict(x) 
            yy     = np.concatenate((y, y_pred), axis = 1) # yy = (y_true_charged, y_true_neutral, y_pred_charged, y_pred_neutral)

            y_true = y

            h_pt_tot.Fill((y_pred[l,2]-y_true[l,2])/y_true[l,2])
            def_h_pt_tot.Fill((z[l,1]-y_true[l,2])/y_true[l,2])

            m = make_sqrt(y_pred[l,3]) - math.sqrt(y_true[l,3])
            h_m2_tot.Fill(m)
            m = z[l,2]- math.sqrt(y_true[l,3])
            def_h_m2_tot.Fill(m)
                
            count_steps += 1
            if count_steps % 1000 == 0: print('Test set at the:',count_steps,'th step')
            if count_steps >= n_steps_test: break
        
        ##################################################################################
        R.gROOT.SetBatch(True)

        c1 = R.TCanvas( 'c1', '', 200, 10, 700, 500)
        legend1 = plot_res(h_pt_tot, def_h_pt_tot, "Relative difference for pt")
        legend1.Draw("SAMES")
        c2 = R.TCanvas( 'c4', '', 200, 10, 700, 500)
        legend2 = plot_res(h_m2_tot, def_h_m2_tot, "Absolute difference for mass [GeV]")
        legend2.Draw("SAMES")
        c1.SaveAs(filename+'h_p4_resolution.pdf[')
        c1.SaveAs(filename+'h_p4_resolution.pdf')
        c2.SaveAs(filename+'h_p4_resolution.pdf')
        c2.SaveAs(filename+'h_p4_resolution.pdf]')


    elif(mode=="dm"):
        conf_dm_mat = None
        conf_dm_mat_old = None
        dm_bins = [-0.5,0.5,1.5,2.5,3.5,9.5,10.5,11.5,12.5,23.5]

        print('\nStart generator loop to predict configuration matrices:\n')
        for x,y,z in generator_xyz(): # y is a (n_tau,n_labels) array

            y_pred = model.predict(x) 
            yy     = np.concatenate((y, y_pred), axis = 1) # yy = (y_true_charged, y_true_neutral, y_pred_charged, y_pred_neutral)

            ## Round charge and neutral cosunt to integers:
            y_r      = np.round(y,0)
            y_pred_r = np.round(y_pred,0)

            h_dm = decay_mode_histo((y_r[:,0]-1)*5 + y_r[:,1], (y_pred_r[:,0]-1)*5 + y_pred_r[:,1], dm_bins) # Decay mode comparison to new reconstruction
            h_dm_old = decay_mode_histo((y_r[:,0]-1)*5 + y_r[:,1], z[:,0], dm_bins) # Decay mode comparison to old reconstruction
            
            if conf_dm_mat is None:
                conf_dm_mat     = h_dm
                conf_dm_mat_old = h_dm_old
            else:
                conf_dm_mat += h_dm
                conf_dm_mat_old += h_dm_old
                
            count_steps += 1
            if count_steps % 1000 == 0: print('Test set at the:',count_steps,'th step')
            if count_steps >= n_steps_test: break

        return conf_dm_mat, conf_dm_mat_old
        
