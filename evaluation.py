import numpy as np
import ROOT as R
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pp
import seaborn as sns
import os
import matplotlib as mpl
mpl.use('Agg')

from mymodel import *
from plotting import plot_res

def make_sqrt(m):
    if(m>=0): 
        m = math.sqrt(m)
    else: 
        m = -math.sqrt(-m)
    return m

def evaluation(mode, filename, epoch_number):
    filename_plots = os.path.join(filename, "Plots_"+str(epoch_number))
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
            "pt_res_rel" : pt_res_rel,
        }
    elif(mode=="p4_dm"):
        custom_objects = {
            "CustomMSE": CustomMSE,
            # "my_cat_dm" : my_cat_dm,
            "my_acc" : my_acc,
            "my_mse_ch"   : my_mse_ch,
            "my_mse_neu"  : my_mse_neu,
            "my_mse_pt"   : my_mse_pt,
            "my_mse_mass" : my_mse_mass,
            "pt_res" : pt_res,
            "m2_res" : m2_res,
            "pt_res_rel" : pt_res_rel,
            # "my_mse_ch_new"   : my_mse_ch_new,
            # "my_mse_neu_new"  : my_mse_neu_new,
            # "quantile_pt" : quantile_pt,
            # "log_cosh_pt" : log_cosh_pt,
            # "log_cosh_mass" : log_cosh_mass,
            # "my_mse_ch_4"   : my_mse_ch_4,
            # "my_mse_neu_4"  : my_mse_neu_4,
            # "my_mse_pt_4"   : my_mse_pt_4,
            # "my_mse_mass_4" : my_mse_mass_4,
        }

    model = tf.keras.models.load_model(os.path.join(filename,"my_model_{}".format(epoch_number)), custom_objects=custom_objects, compile=True)
    print("Model loaded.")

    generator_xyz, n_batches = make_generator(entry_start_test, entry_stop_test, z = True)

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

            ### pt_res_rel in function of pt-true:
            if y_true_tot is None:
                y_true_tot = y_true[:,2]
                delta_pt_tot = (y_pred[:,2]-y_true[:,2])/y_true[:,2]
                def_delta_pt_tot = (z[:,1]-y_true[:,2])/y_true[:,2]
            else:
                y_true_tot = tf.concat([y_true_tot,y_true[:,2]], axis = 0)
                delta_pt_tot = tf.concat([delta_pt_tot,(y_pred[:,2]-y_true[:,2])/y_true[:,2]], axis = 0)
                def_delta_pt_tot = tf.concat([def_delta_pt_tot,(z[:,1]-y_true[:,2])/y_true[:,2]], axis = 0)
            ### end pt_res_rel in function of pt_true

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
        ### pt_res_rel in function of pt-true:
        q = np.arange(0,100.1,0.1)
        indices = tf.where(tf.logical_and(y_true_tot >= 20, y_true_tot < (20+5))) 
        res_array = tf.gather(delta_pt_tot,indices, axis = 0)
        quantile_array = tpf.stats.percentile(res_array,q, interpolation='linear')
        def_res_array = tf.gather(def_delta_pt_tot,indices, axis = 0)
        def_quantile_array = tpf.stats.percentile(def_res_array,q, interpolation='linear')

        diff_array_0 = tf.math.abs(quantile_array[680]-quantile_array[0])
        i_0 = tf.constant(1)
        c = lambda i, diff_array: i < len(quantile_array)-680
        b = lambda i, diff_array: (i+1, myconcat(diff_array, tf.math.abs(quantile_array[i+680]-quantile_array[i]),i))
        r = tf.while_loop(c, b, [i_0, diff_array_0], shape_invariants=[i_0.get_shape(), tf.TensorShape([None, ])])
        diff_array = r[1]

        pt_res_true_0 = tf.math.reduce_min(diff_array)
        i_0 = tf.constant(25.0)
        j_0 = tf.constant(1)
        c = lambda i, p, j: i < 100
        b = lambda i, p, j: (i+5, mytruept(y_true_tot,delta_pt_tot,p,i,j),j+1)
        r = tf.while_loop(c, b, [i_0, pt_res_true_0, j_0], shape_invariants=[i_0.get_shape(), tf.TensorShape([None,]), j_0.get_shape()])
        pt_res_true_tot = r[1]

        diff_array_0 = tf.math.abs(def_quantile_array[680]-def_quantile_array[0])
        i_0 = tf.constant(1)
        c = lambda i, diff_array: i < len(def_quantile_array)-680
        b = lambda i, diff_array: (i+1, myconcat(diff_array, tf.math.abs(def_quantile_array[i+680]-def_quantile_array[i]),i))
        r = tf.while_loop(c, b, [i_0, diff_array_0], shape_invariants=[i_0.get_shape(), tf.TensorShape([None, ])])
        diff_array = r[1]

        def_pt_res_true_0 = tf.math.reduce_min(diff_array)
        i_0 = tf.constant(25.0)
        j_0 = tf.constant(1)
        c = lambda i, p, j: i < 100
        b = lambda i, p, j: (i+5, mytruept(y_true_tot,def_delta_pt_tot,p,i,j),j+1)
        r = tf.while_loop(c, b, [i_0, def_pt_res_true_0, j_0], shape_invariants=[i_0.get_shape(), tf.TensorShape([None,]), j_0.get_shape()])
        def_pt_res_true_tot = r[1]
        ### end pt_res_rel in function of pt-true

        R.gROOT.SetBatch(True)

        c1 = R.TCanvas( 'c1', '', 200, 10, 700, 500)
        legend1 = plot_res(h_pt_tot, def_h_pt_tot, "Relative difference for pt")
        legend1.Draw("SAMES")
        c2 = R.TCanvas( 'c4', '', 200, 10, 700, 500)
        legend2 = plot_res(h_m2_tot, def_h_m2_tot, "Absolute difference for mass [GeV]")
        legend2.Draw("SAMES")
        c1.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf['))
        c1.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c2.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c2.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf]'))

        x = np.arange(22.5,102.5,5)
        fig0, axes = plt.subplots(1, sharex=False, figsize=(12, 8))
        axes.tick_params(axis='both', which='major', labelsize=14)
        axes.plot(x, pt_res_true_tot, 'bo:', label='predicted')
        axes.plot(x,def_pt_res_true_tot,'ro:', label='default')
        plt.xlabel('pt true [GeV]', fontsize=16)
        plt.ylabel('relative pt resolution', fontsize=16)
        axes.legend(fontsize=16)

        pdf0 = pp.PdfPages(os.path.join(filename_plots,"pt_res_true.pdf"))
        pdf0.savefig(fig0)
        pdf0.close()
        plt.close()


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
    
    elif(mode=="p4_dm"):
        conf_dm_mat = None
        conf_dm_mat_old = None
        dm_bins = [-0.5,0.5,1.5,2.5,3.5,9.5,10.5,11.5,12.5,23.5]
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

        ### quantile part:
        delta_pt = None
        def_delta_pt = None
        ### end quantile part

        ### pt-res in function of pt_true part:
        y_true_tot = None
        delta_pt_tot = None
        def_delta_pt_tot = None
        ### end of pt-res in function of pt_true part

        print('\nStart generator loop to predict:\n')

        for x,y,z in generator_xyz(): # y is a (n_tau,n_labels) array
            # print('Test set at the:',count_steps,'th event')
            y_pred = model.predict(x) 
            yy     = np.concatenate((y, y_pred), axis = 1) # yy = (y_true_charged, y_true_neutral, y_pred_charged, y_pred_neutral)

            ### Round charge and neutral cosunt to integers:
            y_r      = np.round(y,0)
            y_pred_r = np.round(y_pred,0)

            ### Decay mode comparison to new reconstruction:
            h_dm = decay_mode_histo((y_r[:,0]-1)*5 + y_r[:,1], (y_pred_r[:,0]-1)*5 + y_pred_r[:,1], dm_bins)
            
            ##### dm 6 outputs part:
            # dm = tf.constant([0,1,2,10,11,5])
            # y_shape= tf.shape(y_pred_r)
            # dm_mode = None
            # for i in range(y_shape[0]):
            #     indice = tf.math.argmax(y_pred_r[i,:6])
            #     if dm_mode is None:
            #         dm_mode = dm[indice]
            #     else:
            #         a = tf.reshape(dm[indice],(1,))
            #         # print('a: ',a)
            #         dm_mode = tf.reshape(dm_mode,(i,))
            #         # print('dm_mode: ',dm_mode)
            #         dm_mode = tf.concat((dm_mode,a), axis = 0)
            
            # h_dm = decay_mode_histo((y_r[:,0]-1)*5 + y_r[:,1], dm_mode, dm_bins)
            #### end of dm 6 output part

            ### Decay mode comparison to old reconstruction:
            h_dm_old = decay_mode_histo((y_r[:,0]-1)*5 + y_r[:,1], z[:,0], dm_bins)

            ### p4 part:
            y_true = y
        
            l = 0
            true_dm = (y_true[:,0]  -1)*5 + y_true[:,1]
            pred_dm = (y_pred_r[:,0]-1)*5 + y_pred_r[:,1]
            def_dm  = z[:,0]


            ### pt_res_rel in function of pt-true:
            if y_true_tot is None:
                y_true_tot = y_true[:,2]
                delta_pt_tot = (y_pred[:,2]-y_true[:,2])/y_true[:,2]
                def_delta_pt_tot = (z[:,1]-y_true[:,2])/y_true[:,2]
            else:
                y_true_tot = tf.concat([y_true_tot,y_true[:,2]], axis = 0)
                delta_pt_tot = tf.concat([delta_pt_tot,(y_pred[:,2]-y_true[:,2])/y_true[:,2]], axis = 0)
                def_delta_pt_tot = tf.concat([def_delta_pt_tot,(z[:,1]-y_true[:,2])/y_true[:,2]], axis = 0)
            ### end pt_res_rel in function of pt_true


            ### Quantile part:
            if delta_pt is None:
                delta_pt = (y_pred[:,2]-y_true[:,2])/y_true[:,2]
                def_delta_pt = (z[:,1]-y_true[:,2])/y_true[:,2]
            else: 
                a = (y_pred[:,2]-y_true[:,2])/y_true[:,2]
                b = (z[:,1]-y_true[:,2])/y_true[:,2]
                delta_pt = tf.concat([delta_pt,a], axis = 0)
                def_delta_pt = tf.concat([def_delta_pt,b], axis = 0)
            ##### end quantile part


            for dm in true_dm:
                h_pt_tot.Fill((y_pred[l,2]-y_true[l,2])/y_true[l,2])
                def_h_pt_tot.Fill((z[l,1]-y_true[l,2])/y_true[l,2])
                
                if(dm==0):
                    m = make_sqrt(y_pred[l,3]) - math.sqrt(y_true[l,3])
                    h_m2_pi_tot.Fill(m)
                    m = z[l,2]- math.sqrt(y_true[l,3])
                    def_h_m2_pi_tot.Fill(m)
                elif(dm==1 or dm==2 or dm==3):
                    m = make_sqrt(y_pred[l,3]) - math.sqrt(y_true[l,3])
                    h_m2_pi_0_tot.Fill(m)
                    m = z[l,2]-  math.sqrt(y_true[l,3])
                    def_h_m2_pi_0_tot.Fill(m)
                elif(dm==10 or dm==11 or dm==12):
                    m = make_sqrt(y_pred[l,3]) - math.sqrt(y_true[l,3])
                    h_m2_3pi_tot.Fill(m)
                    m = z[l,2]-  math.sqrt(y_true[l,3])
                    def_h_m2_3pi_tot.Fill(m)
                
                if(dm == pred_dm[l]):
                    if(dm==0):
                        m = make_sqrt(y_pred[l,3]) - math.sqrt(y_true[l,3])
                        c_h_m2_pi_tot.Fill(m)
                    elif(dm==1 or dm==2 or dm==3):
                        m = make_sqrt(y_pred[l,3]) - math.sqrt(y_true[l,3])
                        c_h_m2_pi_0_tot.Fill(m)
                    elif(dm==10 or dm==11 or dm==12):
                        m = make_sqrt(y_pred[l,3]) - math.sqrt(y_true[l,3])
                        c_h_m2_3pi_tot.Fill(m)
                
                if(dm == def_dm[l]):
                    if(dm==0):
                        m = z[l,2] - math.sqrt(y_true[l,3])
                        c_def_h_m2_pi_tot.Fill(m)
                    elif(dm==1 or dm==2 or dm==3):
                        m = z[l,2] - math.sqrt(y_true[l,3])
                        c_def_h_m2_pi_0_tot.Fill(m)
                    elif(dm==10 or dm==11 or dm==12):
                        m = z[l,2] - math.sqrt(y_true[l,3])
                        c_def_h_m2_3pi_tot.Fill(m)

                l += 1

            if conf_dm_mat is None:
                conf_dm_mat     = h_dm
                conf_dm_mat_old = h_dm_old
            else:
                conf_dm_mat += h_dm
                conf_dm_mat_old += h_dm_old
                
            count_steps += 1
            if count_steps % 100 == 0: print('Test set at the:',count_steps,'th event')
            if count_steps >= n_steps_test: break
        ################### end of generator loop ##########################


        ### Quantile part:
        q = np.arange(0,101,1)

        quantile_array = tpf.stats.percentile(delta_pt,q, interpolation='linear')
        def_quantile_array = tpf.stats.percentile(def_delta_pt,q, interpolation='linear')

        np.savetxt('quantile_evaluation.csv', [quantile_array,def_quantile_array], delimiter=',')
        ### end quantile part

        ### pt_res_rel in function of pt-true:
        q = np.arange(0,100.1,0.1)
        indices = tf.where(tf.logical_and(y_true_tot >= 20, y_true_tot < (20+5))) 
        res_array = tf.gather(delta_pt_tot,indices, axis = 0)
        quantile_array = tpf.stats.percentile(res_array,q, interpolation='linear')
        def_res_array = tf.gather(def_delta_pt_tot,indices, axis = 0)
        def_quantile_array = tpf.stats.percentile(def_res_array,q, interpolation='linear')

        diff_array_0 = tf.math.abs(quantile_array[680]-quantile_array[0])
        i_0 = tf.constant(1)
        c = lambda i, diff_array: i < len(quantile_array)-680
        b = lambda i, diff_array: (i+1, myconcat(diff_array, tf.math.abs(quantile_array[i+680]-quantile_array[i]),i))
        r = tf.while_loop(c, b, [i_0, diff_array_0], shape_invariants=[i_0.get_shape(), tf.TensorShape([None, ])])
        diff_array = r[1]

        pt_res_true_0 = tf.math.reduce_min(diff_array)
        i_0 = tf.constant(25.0)
        j_0 = tf.constant(1)
        c = lambda i, p, j: i < 100
        b = lambda i, p, j: (i+5, mytruept(y_true_tot,delta_pt_tot,p,i,j),j+1)
        r = tf.while_loop(c, b, [i_0, pt_res_true_0, j_0], shape_invariants=[i_0.get_shape(), tf.TensorShape([None,]), j_0.get_shape()])
        pt_res_true_tot = r[1]

        diff_array_0 = tf.math.abs(def_quantile_array[680]-def_quantile_array[0])
        i_0 = tf.constant(1)
        c = lambda i, diff_array: i < len(def_quantile_array)-680
        b = lambda i, diff_array: (i+1, myconcat(diff_array, tf.math.abs(def_quantile_array[i+680]-def_quantile_array[i]),i))
        r = tf.while_loop(c, b, [i_0, diff_array_0], shape_invariants=[i_0.get_shape(), tf.TensorShape([None, ])])
        diff_array = r[1]

        def_pt_res_true_0 = tf.math.reduce_min(diff_array)
        i_0 = tf.constant(25.0)
        j_0 = tf.constant(1)
        c = lambda i, p, j: i < 100
        b = lambda i, p, j: (i+5, mytruept(y_true_tot,def_delta_pt_tot,p,i,j),j+1)
        r = tf.while_loop(c, b, [i_0, def_pt_res_true_0, j_0], shape_invariants=[i_0.get_shape(), tf.TensorShape([None,]), j_0.get_shape()])
        def_pt_res_true_tot = r[1]
        ### end pt_res_rel in function of pt-true

        ##################################################################################
        R.gROOT.SetBatch(True)

        ## Save pt histograms to a root file:
        # f = R.TFile.Open("pt_histograms.root","RECREATE")
        # h_pt_tot.Write()
        # def_h_pt_tot.Write()
        # f.Close()

        ### Make and save figures:
        c1 = R.TCanvas( 'c1', '', 200, 10, 700, 500)
        legend1 = plot_res(h_pt_tot, def_h_pt_tot, "Relative difference for pt")
        legend1.Draw("SAMES")
        c2 = R.TCanvas( 'c2', '', 200, 10, 700, 500)
        legend2 = plot_res(h_m2_pi_tot, def_h_m2_pi_tot, "Absolute difference for mass for (\pi\pm) [GeV]")
        legend2.Draw("SAMES")
        c3 = R.TCanvas( 'c3', '', 200, 10, 700, 500)
        legend3 = plot_res(h_m2_pi_0_tot, def_h_m2_pi_0_tot, "Absolute difference for mass for (\pi\pm + n \pi0) [GeV]")
        legend3.Draw("SAMES")
        c4 = R.TCanvas( 'c4', '', 200, 10, 700, 500)
        legend4 = plot_res(h_m2_3pi_tot, def_h_m2_3pi_tot, "Absolute difference for mass for (3\pi\pm + n \pi0) [GeV]")
        legend4.Draw("SAMES")
        c5 = R.TCanvas( 'c5', '', 200, 10, 700, 500)
        legend5 = plot_res(c_h_m2_pi_tot, c_def_h_m2_pi_tot, "Absolute difference for mass for (\pi\pm) [GeV]", True)
        legend5.Draw("SAMES")
        c6 = R.TCanvas( 'c6', '', 200, 10, 700, 500)
        legend6 = plot_res(c_h_m2_pi_0_tot, c_def_h_m2_pi_0_tot, "Absolute difference for mass for (\pi\pm + n \pi0) [GeV]", True)
        legend6.Draw("SAMES")
        c7 = R.TCanvas( 'c7', '', 200, 10, 700, 500)
        legend7 = plot_res(c_h_m2_3pi_tot, c_def_h_m2_3pi_tot, "Absolute difference for mass for (3\pi\pm + n \pi0) [GeV]", True)
        legend7.Draw("SAMES")

        c11 = R.TCanvas( 'c11', '', 200, 10, 700, 500)
        legend11 = plot_res(h_pt_tot, def_h_pt_tot, "Relative difference for pt", c_dm = False, logy = True)
        legend11.Draw("SAMES")
        c22 = R.TCanvas( 'c22', '', 200, 10, 700, 500)
        legend22 = plot_res(h_m2_pi_tot, def_h_m2_pi_tot, "Absolute difference for mass for (\pi\pm) [GeV]", c_dm = False, logy = True)
        legend22.Draw("SAMES")
        c33 = R.TCanvas( 'c33', '', 200, 10, 700, 500)
        legend33 = plot_res(h_m2_pi_0_tot, def_h_m2_pi_0_tot, "Absolute difference for mass for (\pi\pm + n \pi0) [GeV]", c_dm = False, logy = True)
        legend33.Draw("SAMES")
        c44 = R.TCanvas( 'c44', '', 200, 10, 700, 500)
        legend44 = plot_res(h_m2_3pi_tot, def_h_m2_3pi_tot, "Absolute difference for mass for (3\pi\pm + n \pi0) [GeV]", c_dm = False, logy = True)
        legend44.Draw("SAMES")
        c55 = R.TCanvas( 'c55', '', 200, 10, 700, 500)
        legend55 = plot_res(c_h_m2_pi_tot, c_def_h_m2_pi_tot, "Absolute difference for mass for (\pi\pm) [GeV]", c_dm = True, logy = True)
        legend55.Draw("SAMES")
        c66 = R.TCanvas( 'c66', '', 200, 10, 700, 500)
        legend66 = plot_res(c_h_m2_pi_0_tot, c_def_h_m2_pi_0_tot, "Absolute difference for mass for (\pi\pm + n \pi0) [GeV]", c_dm = True, logy = True)
        legend66.Draw("SAMES")
        c77 = R.TCanvas( 'c77', '', 200, 10, 700, 500)
        legend77 = plot_res(c_h_m2_3pi_tot, c_def_h_m2_3pi_tot, "Absolute difference for mass for (3\pi\pm + n \pi0) [GeV]", c_dm = True, logy = True)
        legend77.Draw("SAMES")

        c1.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf['))
        c1.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c2.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c3.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c4.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c5.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c6.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c7.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c11.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c22.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c33.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c44.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c55.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c66.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c77.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf'))
        c77.SaveAs(os.path.join(filename_plots,'h_p4_resolution.pdf]'))

        x = np.arange(22.5,102.5,5)
        fig0, axes = plt.subplots(1, sharex=False, figsize=(12, 8))
        axes.tick_params(axis='both', which='major', labelsize=14)
        axes.plot(x, pt_res_true_tot, 'bo:', label='predicted')
        axes.plot(x,def_pt_res_true_tot,'ro:', label='default')
        plt.xlabel('pt true [GeV]', fontsize=16)
        plt.ylabel('relative pt resolution', fontsize=16)
        axes.legend(fontsize=16)

        pdf0 = pp.PdfPages(os.path.join(filename_plots,"pt_res_true.pdf"))
        pdf0.savefig(fig0)
        pdf0.close()
        plt.close()
        
        return conf_dm_mat, conf_dm_mat_old



def myconcat(a,b,i):
    a = tf.reshape(a,(i,))
    b = tf.reshape(b,(1,))
    a = tf.concat([a, b], axis = 0)
    return a

def mytruept(y_true_tot, delta_pt_tot, pt_res_true, i, j):
    indices = tf.where(tf.logical_and(y_true_tot >= i, y_true_tot < (i+5))) 
    res_array = tf.gather(delta_pt_tot,indices, axis = 0) # array of res_pt_rel in this pt_true interval

    ### Build quantiles in 0.1:
    counter = 0
    diff_array = None
    q = np.arange(0,100.1,0.1)
    quantile_array = tpf.stats.percentile(res_array,q, interpolation='linear')

    diff_array_0 = tf.math.abs(quantile_array[680]-quantile_array[0])
    i_0 = tf.constant(1)
    c = lambda i, diff_array: i < len(quantile_array)-680
    b = lambda i, diff_array: (i+1, myconcat(diff_array, tf.math.abs(quantile_array[i+680]-quantile_array[i]),i))
    r = tf.while_loop(c, b, [i_0, diff_array_0], shape_invariants=[i_0.get_shape(), tf.TensorShape([None, ])])
    diff_array = r[1]

    a = tf.math.reduce_min(diff_array)
    a = tf.reshape(a,(1,))
    b = pt_res_true
    b = tf.reshape(b,(j,))
    pt_res_true = tf.concat([b,a], axis = 0)
    return pt_res_true