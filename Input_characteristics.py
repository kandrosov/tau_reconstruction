import ROOT as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

df = R.RDataFrame('taus','/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root')

feature_names = ["pfCand_pt", "pfCand_eta", "pfCand_phi", "pfCand_mass", "pfCand_pdgId", "pfCand_charge", 
                "pfCand_pvAssociationQuality", "pfCand_fromPV", "pfCand_puppiWeight", "pfCand_puppiWeightNoLep", "pfCand_lostInnerHits", 
                "pfCand_numberOfPixelHits", "pfCand_numberOfHits", "pfCand_hasTrackDetails", "pfCand_dxy", "pfCand_dxy_error", "pfCand_dz", 
                "pfCand_dz_error", "pfCand_track_chi2", "pfCand_track_ndof", "pfCand_caloFraction", "pfCand_hcalFraction", 
                "pfCand_rawCaloFraction", "pfCand_rawHcalFraction"]

scale_list    = np.array([1,2,20,21,22,23])
stdscale_list = np.array([0,3,12,14,15,16,17,18,19])

data_scale = {}
data_stdscale = {}

for i in scale_list:
    print(feature_names[i])
    h     = df.Histo1D(feature_names[i])
    mymax = h.GetMaximum()
    mymin = h.GetMinimum()
    data_scale[feature_names[i]] = []
    data_scale[feature_names[i]].append({
        'name': feature_names[i],
        'max' : mymax,
        'min' : mymin,
    })

for i in stdscale_list:
    print(feature_names[i])
    h     = df.Histo1D(feature_names[i])
    mean  = h.GetMean()
    std   = h.GetStdDev()
    data_stdscale[feature_names[i]] = []
    data_stdscale[feature_names[i]].append({
        'name': feature_names[i],
        'std' : std,
        'mean': mean
    })

with open('mean_std.txt', 'w') as outfile:
    json.dump(data_stdscale, outfile) # writes the data object to the outfile file
with open('min_max.txt', 'w') as outfile:
    json.dump(data_scale, outfile) # writes the data object to the outfile file