import ROOT as R
import tensorflow as tf
import numpy as np

df = R.RDataFrame('taus','/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root')#.Range(0,1000)

print('DataFrame charged')

df_def = df.Filter('tau_index >= 0 && tau_decayModeFindingNewDMs > 0')

print('Filter applied')

np_def  = df_def.AsNumpy(columns=["tau_decayMode","tau_pt","tau_eta","tau_phi","tau_mass"])
np_true = df_def.AsNumpy(columns=["lepton_gen_vis_pdg","lepton_gen_vis_pt","lepton_gen_vis_eta","lepton_gen_vis_phi","lepton_gen_vis_mass"])

print('Converted to numpy')

l = len(np_true["lepton_gen_vis_pt"])

true_pt   = np.zeros(l)
true_eta  = np.zeros(l)
true_phi  = np.zeros(l)
true_mass = np.zeros(l)

def_n_neu = np.zeros(l)
def_n_ch  = np.zeros(l)

true_n_ch  = np.zeros(l)
true_n_neu = np.zeros(l)

for tau_ind in range(len(np_true["lepton_gen_vis_pt"])):
    ### default n_ch and n_neu from dm
    dm = np_def["tau_decayMode"][tau_ind]
    def_n_ch[tau_ind]  = dm//5 + 1
    def_n_neu[tau_ind] = dm%5

    ### 4-momentum and n_ch, n_neu part for true
    gen_p4 = R.TLorentzVector(0, 0, 0, 0)
    v1     = R.TLorentzVector(0, 0, 0, 0)

    cnt_ch  = 0
    cnt_neu = 0
    cnt_photons = 0

    for pf_ind in range(len(np_true["lepton_gen_vis_pt"][tau_ind])):
        v1.SetPtEtaPhiM(np_true["lepton_gen_vis_pt"][tau_ind][pf_ind] , np_true["lepton_gen_vis_eta"][tau_ind][pf_ind],
                        np_true["lepton_gen_vis_phi"][tau_ind][pf_ind], np_true["lepton_gen_vis_mass"][tau_ind][pf_ind])
        gen_p4 += v1

        ### n_ch and n_neu for true
        pdg_id = np_true["lepton_gen_vis_pdg"][tau_ind][pf_ind]
        pdg_id = abs(pdg_id)
        if(pdg_id == 22):
            cnt_photons += 1
            if(cnt_photons == 2):
                cnt_neu += 1
                cnt_photons = 0
        elif(pdg_id == 310 or pdg_id == 130):
            cnt_neu += 1
        elif(pdg_id == 211 or pdg_id == 321):
            cnt_ch += 1
    
    true_n_neu[tau_ind] = cnt_neu
    true_n_ch[tau_ind]  = cnt_ch
            
    true_pt[tau_ind]   = gen_p4.Pt()
    true_eta[tau_ind]  = gen_p4.Eta()
    true_phi[tau_ind]  = gen_p4.Phi()
    true_mass[tau_ind] = gen_p4.M()*gen_p4.M()

    if(tau_ind%100000==0): print(tau_ind)

mse_ch   = np.mean(np.square(def_n_ch  - true_n_ch))
mse_neu  = np.mean(np.square(def_n_neu - true_n_neu))
mse_pt   = np.mean(np.square((np_def["tau_pt"]  - true_pt)/true_pt))
mse_eta  = np.mean(np.square(np_def["tau_eta"] - true_eta))
mse_phi  = np.mean(np.square(np_def["tau_phi"] - true_phi))
mse_mass = np.mean(np.square(np_def["tau_mass"]*np_def["tau_mass"]- true_mass))
print('\n\nmse_ch:   ', mse_ch)
print('mse_neu:  ', mse_neu)
print('mse_pt:   ', mse_pt)
print('mse_eta:  ', mse_eta)
print('mse_phi:  ', mse_phi)
print('mse_mass: ', mse_mass, '\n\n')
