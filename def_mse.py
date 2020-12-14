import ROOT as R
import tensorflow as tf
import numpy as np

R.gInterpreter.Declare('''
TLorentzVector GetGenP4(const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
                        const ROOT::VecOps::RVec<float>& phi, const ROOT::VecOps::RVec<float>& mass)
{
    TLorentzVector total_p4(0, 0, 0, 0);
    for(size_t n = 0; n < pt.size(); ++n) {
        TLorentzVector p4;
        p4.SetPtEtaPhiM(pt.at(n), eta.at(n), phi.at(n), mass.at(n));
        total_p4 += p4;
    }
    return total_p4;
}

// Function that counts the true values for the number of charged hadrons for the event:
int Count_charged_hadrons_true(const ROOT::VecOps::RVec<int>& lepton_gen_vis_pdg){
    size_t cnt_charged_hadrons = 0;
    for(int pdg_id : lepton_gen_vis_pdg) {
        pdg_id = std::abs(pdg_id);
        if(pdg_id == 211 || pdg_id == 321) {
            ++cnt_charged_hadrons;
        }
    }
    return cnt_charged_hadrons;
}

// Function that counts the true values for the number of neutral hadrons for the event:
int Count_neutral_hadrons_true(const ROOT::VecOps::RVec<int>& lepton_gen_vis_pdg){
    size_t cnt_neutral_hadrons = 0;
    size_t cnt_photons = 0;
    for(int pdg_id : lepton_gen_vis_pdg) {
        pdg_id = std::abs(pdg_id);
        // 2 photons = 1 pi0:
        if(pdg_id == 22){
            ++cnt_photons;
            if(cnt_photons == 2){
                ++cnt_neutral_hadrons;
                cnt_photons = 0;
            }
        }else if(pdg_id == 310 || pdg_id == 130){
            ++cnt_neutral_hadrons;
        }
    }
    return cnt_neutral_hadrons;
}

'''
)

df = R.RDataFrame('taus','/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root')#.Range(0,1000)

print('DataFrame charged')

df = df.Filter('tau_index >= 0 && tau_decayModeFindingNewDMs > 0')

df = df.Define('gen_p4', 'GetGenP4(lepton_gen_vis_pt, lepton_gen_vis_eta, lepton_gen_vis_phi, lepton_gen_vis_mass)') \
       .Define('gen_pt', 'gen_p4.Pt()').Define('gen_eta', 'gen_p4.Eta()').Define('gen_phi', 'gen_p4.Phi()') \
       .Define('gen_mass', 'gen_p4.M()') \
       .Define('pt_rel', '(tau_pt - gen_pt) / gen_pt') \
       .Define('se_pt_rel', 'pt_rel * pt_rel') \
       .Define('abs_pt_rel', 'std::abs(pt_rel)') \
       .Define('lc_pt_rel', 'std::log(std::cosh(pt_rel))') \
       .Define('delta_m2', 'tau_mass * tau_mass - gen_mass * gen_mass') \
       .Define('abs_delta_m2', 'std::abs(delta_m2)') \
       .Define('se_m2', 'delta_m2 * delta_m2') \
       .Define('tau_n_ch', 'int(tau_decayMode / 5 + 1)') \
       .Define('tau_n_ne', 'int(tau_decayMode % 5)') \
       .Define('gen_n_ch', 'Count_charged_hadrons_true(lepton_gen_vis_pdg)') \
       .Define('gen_n_ne', 'Count_neutral_hadrons_true(lepton_gen_vis_pdg)') \
       .Define('delta_ch', 'tau_n_ch - gen_n_ch') \
       .Define('delta_ch2', 'delta_ch * delta_ch') \
       .Define('delta_ne', 'tau_n_ne - gen_n_ne') \
       .Define('delta_ne2', 'delta_ne * delta_ne') \


df = df.Filter('gen_pt > 15')
counting = df.Count()
print('Number of entries: ', counting.GetValue()) #14138155

mse_pt_rel = df.Mean('se_pt_rel')
mse_m2 = df.Mean('se_m2')
abs_pt_rel = df.Mean('abs_pt_rel')
abs_delta_m2 = df.Mean('abs_delta_m2')
lc_pt_rel = df.Mean('lc_pt_rel')
std_pt_rel = df.StdDev('pt_rel')
std_m2 = df.StdDev('delta_m2')
mse_delta_ch = df.Mean('delta_ch2')
mse_delta_ne = df.Mean('delta_ne2')

gen_pt_min = df.Min('gen_pt')

print('Filter applied')

print("gen_pt_min = {}".format(gen_pt_min.GetValue()))
print("mse_pt_rel = {}".format(mse_pt_rel.GetValue()))
print("abs_pt_rel = {}".format(abs_pt_rel.GetValue()))
print("lc_pt_rel = {}".format(lc_pt_rel.GetValue()))
print("mse_m2 = {}".format(mse_m2.GetValue()))
print("abs_delta_m2 = {}".format(abs_delta_m2.GetValue()))
print("std_pt_rel = {}".format(std_pt_rel.GetValue()))
print("std_m2 = {}".format(std_m2.GetValue()))
print("mse_delta_ch = {}".format(mse_delta_ch.GetValue()))
print("mse_delta_ne = {}".format(mse_delta_ne.GetValue()))
