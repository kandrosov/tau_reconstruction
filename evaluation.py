import numpy as np
import ROOT as R
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pp
import seaborn as sns

from mymodel_copy import *
from plotting import plot_res

def make_sqrt(m):
    if(m>=0): 
        m = math.sqrt(m)
    else: 
        m = -math.sqrt(-m)
    return m

def evaluation():
    print('\nStart evaluation, load model and generator:\n')
    ##### Load the model:
    custom_objects = {
        "CustomMSE": CustomMSE,
        "my_acc" : my_acc,
        "pt_res" : pt_res,
        "eta_res": eta_res,
        "phi_res": phi_res,
        "m2_res" : m2_res,
    }
    model = tf.keras.models.load_model("/data/cedrine/Models0/my_model_{}".format(2), custom_objects=custom_objects, compile=True)
    print("Model loaded.")

    generator_xyz, n_batches = make_generator('/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root',entry_start_test, entry_stop_test, z = True)

    conf_dm_mat = None
    conf_dm_mat_old = None
    dm_bins = [-0.5,0.5,1.5,2.5,3.5,9.5,10.5,11.5,12.5,23.5]
    count_steps = 0

    h_pt_tot      = R.TH1F('h_pt'        , 'pt resolution'  , 200, -2, 1.5)#nbins, xlow, xup
    h_eta_tot     = R.TH1F('h_eta'       , 'eta resolution' , 200, -0.2, 0.2)
    h_phi_tot     = R.TH1F('h_phi'       , 'phi resolution' , 200, -0.2, 0.2)
    h_m2_pi_tot   = R.TH1F('h_m2_pi'     , 'm2 1 resolution', 200, -1, 1)
    h_m2_pi_0_tot = R.TH1F('h_m2_pi_pi0' , 'm2 2 resolution', 200, -3, 2)
    h_m2_3pi_tot  = R.TH1F('h_m2_3pi_pi0', 'm2 3 resolution', 200, -3, 2)

    def_h_pt_tot      = R.TH1F('def_h_pt'        , 'default pt resolution'  , 200, -2, 1.5)
    def_h_eta_tot     = R.TH1F('def_h_eta'       , 'default eta resolution' , 200, -0.2, 0.2)
    def_h_phi_tot     = R.TH1F('def_h_phi'       , 'default phi resolution' , 200, -0.2, 0.2)
    def_h_m2_pi_tot   = R.TH1F('def_h_m2_pi'     , 'default m2 1 resolution', 200, -1, 1)
    def_h_m2_pi_0_tot = R.TH1F('def_h_m2_pi_pi0' , 'default m2 2 resolution', 200, -3, 2)
    def_h_m2_3pi_tot  = R.TH1F('def_h_m2_3pi_pi0', 'default m2 3 resolution', 200, -3, 2)

    c_h_m2_pi_tot   = R.TH1F('c_h_m2_pi'     , 'c m2 1 resolution', 200, -1, 1)
    c_h_m2_pi_0_tot = R.TH1F('c_h_m2_pi_pi0' , 'c m2 2 resolution', 200, -3, 2)
    c_h_m2_3pi_tot  = R.TH1F('c_h_m2_3pi_pi0', 'c m2 3 resolution', 200, -3, 2)

    c_def_h_m2_pi_tot   = R.TH1F('c_def_h_m2_pi'     , 'c default m2 1 resolution', 200, -1, 1)
    c_def_h_m2_pi_0_tot = R.TH1F('c_def_h_m2_pi_pi0' , 'c default m2 2 resolution', 200, -3, 2)
    c_def_h_m2_3pi_tot  = R.TH1F('c_def_h_m2_3pi_pi0', 'c default m2 3 resolution', 200, -3, 2)

    c_true_m = R.TH1F('c_true_m', 'test true', 200, -2, 4)
    c_def_m  = R.TH1F('c_def_m', 'test def'  , 200, -2, 4)

    print('\nStart generator loop to predict:\n')

    for x,y,z in generator_xyz(): # y is a (n_tau,n_labels) array
        y_pred = model.predict(x) 
        yy     = np.concatenate((y, y_pred), axis = 1) # yy = (y_true_charged, y_true_neutral, y_pred_charged, y_pred_neutral)

        ### Round charge and neutral cosunt to integers:
        y_r      = np.round(y,0)
        y_pred_r = np.round(y_pred,0)

        ### Decay mode comparison to new reconstruction:
        h_dm = decay_mode_histo((y_r[:,0]-1)*5 + y_r[:,1], (y_pred_r[:,0]-1)*5 + y_pred_r[:,1], dm_bins)

        ### Decay mode comparison to old reconstruction:
        h_dm_old = decay_mode_histo((y_r[:,0]-1)*5 + y_r[:,1], z[:,0], dm_bins)

        ### p4 part:
        y_true = y
       
        l = 0
        true_dm = (y_true[:,0]  -1)*5 + y_true[:,1]
        pred_dm = (y_pred_r[:,0]-1)*5 + y_pred_r[:,1]
        def_dm  = z[:,0]

        for dm in true_dm:
            h_pt_tot.Fill((y_pred[l,2]-y_true[l,2])/y_true[l,2])
            h_eta_tot.Fill(y_pred[l,3]-y_true[l,3])
            h_phi_tot.Fill(y_pred[l,4]-y_true[l,4])
            def_h_pt_tot.Fill((z[l,1]-y_true[l,2])/y_true[l,2])
            def_h_eta_tot.Fill(z[l,2]-y_true[l,3])
            def_h_phi_tot.Fill(z[l,3]-y_true[l,4])
            
            if(dm==0):
                m = make_sqrt(y_pred[l,5]) - math.sqrt(y_true[l,5])
                h_m2_pi_tot.Fill(m)
                m = z[l,4]- math.sqrt(y_true[l,5])
                def_h_m2_pi_tot.Fill(m)
            elif(dm==1 or dm==2 or dm==3):
                m = make_sqrt(y_pred[l,5]) - math.sqrt(y_true[l,5])
                h_m2_pi_0_tot.Fill(m)
                m = z[l,4]-  math.sqrt(y_true[l,5])
                def_h_m2_pi_0_tot.Fill(m)
            elif(dm==10 or dm==11 or dm==12):
                m = make_sqrt(y_pred[l,5]) - math.sqrt(y_true[l,5])
                h_m2_3pi_tot.Fill(m)
                m = z[l,4]-  math.sqrt(y_true[l,5])
                def_h_m2_3pi_tot.Fill(m)
            
            if(dm == pred_dm[l]):
                if(dm==0):
                    m = make_sqrt(y_pred[l,5]) - math.sqrt(y_true[l,5])
                    c_h_m2_pi_tot.Fill(m)
                elif(dm==1 or dm==2 or dm==3):
                    m = make_sqrt(y_pred[l,5]) - math.sqrt(y_true[l,5])
                    c_h_m2_pi_0_tot.Fill(m)
                elif(dm==10 or dm==11 or dm==12):
                    m = make_sqrt(y_pred[l,5]) - math.sqrt(y_true[l,5])
                    c_h_m2_3pi_tot.Fill(m)
            
            if(dm == def_dm[l]):
                if(dm==0):
                    m = z[l,4] - math.sqrt(y_true[l,5])
                    c_def_h_m2_pi_tot.Fill(m)
                elif(dm==1 or dm==2 or dm==3):
                    m = z[l,4] - math.sqrt(y_true[l,5])
                    c_def_h_m2_pi_0_tot.Fill(m)
                elif(dm==10 or dm==11 or dm==12):
                    m = z[l,4] - math.sqrt(y_true[l,5])
                    c_def_h_m2_3pi_tot.Fill(m)

            l += 1

        if conf_dm_mat is None:
            conf_dm_mat     = h_dm
            conf_dm_mat_old = h_dm_old
        else:
            conf_dm_mat += h_dm
            conf_dm_mat_old += h_dm_old
            
        count_steps += 1
        if count_steps % 1000 == 0: print('Test set at the:',count_steps,'th event')
        if count_steps >= n_steps_test: break

    print('#######################################################\
      \n            Evaluation finished !!!                  \n\
#######################################################')

    ##################################################################################
    R.gROOT.SetBatch(True)

    c1 = R.TCanvas( 'c1', '', 200, 10, 700, 500)
    legend1 = plot_res(h_pt_tot, def_h_pt_tot, "Relative difference for pt")
    legend1.Draw("SAMES")
    c2 = R.TCanvas( 'c2', '', 200, 10, 700, 500)
    legend2 = plot_res(h_eta_tot, def_h_eta_tot, "Absolute difference for eta")
    legend2.Draw("SAMES")
    c3 = R.TCanvas( 'c3', '', 200, 10, 700, 500)
    legend3 = plot_res(h_phi_tot, def_h_phi_tot, "Absolute difference for phi")
    legend3.Draw("SAMES")
    c4 = R.TCanvas( 'c4', '', 200, 10, 700, 500)
    legend4 = plot_res(h_m2_pi_tot, def_h_m2_pi_tot, "Absolute difference for mass for (\pi\pm) [GeV]")
    legend4.Draw("SAMES")
    c5 = R.TCanvas( 'c5', '', 200, 10, 700, 500)
    legend5 = plot_res(h_m2_pi_0_tot, def_h_m2_pi_0_tot, "Absolute difference for mass for (\pi\pm + n \pi0) [GeV]")
    legend5.Draw("SAMES")
    c6 = R.TCanvas( 'c6', '', 200, 10, 700, 500)
    legend6 = plot_res(h_m2_3pi_tot, def_h_m2_3pi_tot, "Absolute difference for mass for (3\pi\pm + n \pi0) [GeV]")
    legend6.Draw("SAMES")
    c7 = R.TCanvas( 'c7', '', 200, 10, 700, 500)
    legend7 = plot_res(c_h_m2_pi_tot, c_def_h_m2_pi_tot, "Absolute difference for mass for (\pi\pm) [GeV]", True)
    legend7.Draw("SAMES")
    c8 = R.TCanvas( 'c8', '', 200, 10, 700, 500)
    legend8 = plot_res(c_h_m2_pi_0_tot, c_def_h_m2_pi_0_tot, "Absolute difference for mass for (\pi\pm + n \pi0) [GeV]", True)
    legend8.Draw("SAMES")
    c9 = R.TCanvas( 'c9', '', 200, 10, 700, 500)
    legend9 = plot_res(c_h_m2_3pi_tot, c_def_h_m2_3pi_tot, "Absolute difference for mass for (3\pi\pm + n \pi0) [GeV]", True)
    legend9.Draw("SAMES")
    c1.SaveAs('../Plots/h_p4_resolution.pdf[')
    c1.SaveAs('../Plots/h_p4_resolution.pdf')
    c2.SaveAs('../Plots/h_p4_resolution.pdf')
    c3.SaveAs('../Plots/h_p4_resolution.pdf')
    c4.SaveAs('../Plots/h_p4_resolution.pdf')
    c5.SaveAs('../Plots/h_p4_resolution.pdf')
    c6.SaveAs('../Plots/h_p4_resolution.pdf')
    c7.SaveAs('../Plots/h_p4_resolution.pdf')
    c8.SaveAs('../Plots/h_p4_resolution.pdf')
    c9.SaveAs('../Plots/h_p4_resolution.pdf')
    c9.SaveAs('../Plots/h_p4_resolution.pdf]')
    
    return conf_dm_mat, conf_dm_mat_old
