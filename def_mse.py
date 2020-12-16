import ROOT as R
# import tensorflow as tf
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

df = R.RDataFrame('taus','/data/store/reco_skim_v1/tau_DYJetsToLL_M-50_v2.root')#.Range(0,1000)

print('DataFrame charged')

df = df.Filter('tau_index >= 0 && tau_decayModeFindingNewDMs > 0')

df = df.Define('gen_p4_', 'GetGenP4(lepton_gen_vis_pt, lepton_gen_vis_eta, lepton_gen_vis_phi, lepton_gen_vis_mass)') \
        .Define('gen_pt', 'gen_p4_.Pt()').Define('gen_eta', 'gen_p4_.Eta()').Define('gen_phi', 'gen_p4_.Phi()') \
        .Define('pt_rel', '(tau_pt - gen_pt) / gen_pt')

df = df.Filter('pt_rel <= 1')

df = df.Define('gen_mass', 'gen_p4_.M()') \
    .Define('delta_m2', 'tau_mass * tau_mass - gen_mass * gen_mass') \

df = df.Filter('delta_m2 <= 3')

df = df.Define('pt_rel_abs', 'std::abs((tau_pt - gen_pt) / gen_pt)') \
       .Define('my_mse_pt_4', 'std::pow(pt_rel,4)') \
       .Define('se_pt_rel', 'pt_rel * pt_rel') \
       .Define('abs_pt_rel', 'std::abs(pt_rel)') \
       .Define('lc_pt_rel', 'std::log(std::cosh(pt_rel*30))') \
       .Define('my_mse_mass_4', 'std::pow(delta_m2,4)') \
       .Define('log_cosh_m2','std::log(std::cosh(delta_m2*30))') \
       .Define('abs_delta_m2', 'std::abs(delta_m2)') \
       .Define('se_m2', 'delta_m2 * delta_m2') \
       .Define('tau_n_ch', 'int(tau_decayMode / 5 + 1)') \
       .Define('tau_n_ne', 'int(tau_decayMode % 5)') \
       .Define('gen_n_ch', 'Count_charged_hadrons_true(lepton_gen_vis_pdg)') \
       .Define('gen_n_ne', 'Count_neutral_hadrons_true(lepton_gen_vis_pdg)') \
       .Define('delta_ch', 'tau_n_ch - gen_n_ch') \
       .Define('my_mse_ch_4', 'std::pow(delta_ch,4)')\
       .Define('delta_ch2', 'delta_ch * delta_ch') \
       .Define('delta_ne', 'tau_n_ne - gen_n_ne') \
       .Define('my_mse_ne_4', 'std::pow(delta_ne,4)')\
       .Define('delta_ne2', 'delta_ne * delta_ne') \


df = df.Filter('gen_pt > 15')
counting = df.Count()
print('Number of entries: ', counting.GetValue()) #14'138'155

np_df  = df.AsNumpy(columns=["pt_rel"])
## the passage to a is needed for it to work
a = np.zeros(len(np_df["pt_rel"]))
a = np_df["pt_rel"]
q60 = np.percentile(a,60., interpolation='linear')
q40 = np.percentile(a,40., interpolation='linear')
delta_q = np.abs(np.mean(q60-q40))

## Quantiles with step 1%
q = np.arange(0,101,1)
# print(q)# da 0 a 100 inclusi
print(len(q)) # = 101
quantile_array = np.zeros(len(q))
for i in q:
    quantile_array[i-1] = np.percentile(a,i, interpolation='linear')
print('Quantiles: ',quantile_array)

# save to csv file
np.savetxt('quantile.csv', quantile_array, delimiter=',')

mse_pt_rel = df.Mean('se_pt_rel')
mse_m2 = df.Mean('se_m2')
abs_pt_rel = df.Mean('abs_pt_rel')
abs_delta_m2 = df.Mean('abs_delta_m2')
lc_pt_rel = df.Mean('lc_pt_rel')
std_pt_rel = df.StdDev('pt_rel')
std_m2 = df.StdDev('delta_m2')
mse_delta_ch = df.Mean('delta_ch2')
mse_delta_ne = df.Mean('delta_ne2')

mean_log_cosh_m2 = df.Mean('log_cosh_m2')
mean_my_mse_pt_4 = df.Mean('my_mse_pt_4')
mean_my_mse_mass_4 = df.Mean('my_mse_mass_4')
mean_my_mse_ch_4 = df.Mean('my_mse_ch_4')
mean_my_mse_ne_4 = df.Mean('my_mse_ne_4')
mean_pt_rel_abs = df.Mean('pt_rel_abs')

gen_pt_min = df.Min('gen_pt')
pt_rel_min = df.Min('pt_rel')
pt_rel_max = df.Max('pt_rel')

delta_m2_min = df.Min('delta_m2')
delta_m2_max = df.Max('delta_m2')

print('Filter applied')

print("gen_pt_min = {}".format(gen_pt_min.GetValue()))
print("mse_pt_rel = {}".format(mse_pt_rel.GetValue()))
print("abs_pt_rel = {}".format(abs_pt_rel.GetValue()))
print("lc_pt_rel = {}".format(lc_pt_rel.GetValue())) # log_cosh_pt
print("mse_m2 = {}".format(mse_m2.GetValue()))
print("abs_delta_m2 = {}".format(abs_delta_m2.GetValue()))
print("std_pt_rel = {}".format(std_pt_rel.GetValue()))
print("std_m2 = {}".format(std_m2.GetValue()))
print("mse_delta_ch = {}".format(mse_delta_ch.GetValue()))
print("mse_delta_ne = {}".format(mse_delta_ne.GetValue()))

print("mean_log_cosh_m2 = {}".format(mean_log_cosh_m2.GetValue()))
print("mean_my_mse_pt_4 = {}".format(mean_my_mse_pt_4.GetValue()))
print("mean_my_mse_mass_4 = {}".format(mean_my_mse_mass_4.GetValue()))
print("mean_my_mse_ch_4 = {}".format(mean_my_mse_ch_4.GetValue()))
print("mean_my_mse_ne_4 = {}".format(mean_my_mse_ne_4.GetValue()))
print("min pt_rel = {}".format(pt_rel_min.GetValue()))
print("max pt_rel = {}".format(pt_rel_max.GetValue()))
print("min delta_m2 = {}".format(delta_m2_min.GetValue()))
print("max delta_m2 = {}".format(delta_m2_max.GetValue()))

print("delta_q = ", delta_q)
print("pt_rel_abs = {}".format(mean_pt_rel_abs.GetValue()))