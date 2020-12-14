import ROOT as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

df = R.RDataFrame('taus','/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root')
counting = df.Count()
print('Number of entries: ', counting.GetValue())

# feature_names = ["pfCand_pt", "pfCand_eta", "pfCand_phi", "pfCand_mass"]#, "pfCand_pdgId", "pfCand_charge", 
                # "pfCand_pvAssociationQuality", "pfCand_fromPV", "pfCand_puppiWeight", "pfCand_puppiWeightNoLep", "pfCand_lostInnerHits", 
                # "pfCand_numberOfPixelHits", "pfCand_numberOfHits", "pfCand_hasTrackDetails", "pfCand_dxy", "pfCand_dxy_error", "pfCand_dz", 
                # "pfCand_dz_error", "pfCand_track_chi2", "pfCand_track_ndof", "pfCand_caloFraction", "pfCand_hcalFraction", 
                # "pfCand_rawCaloFraction", "pfCand_rawHcalFraction"]

# scale_list    = np.array([1,2,20,21,22,23])
# stdscale_list = np.array([0,3,12,14,15,16,17,18,19])
scale_list    = np.array([0,1,2,11,12,13,14,15,16])
stdscale_list = np.array([3,4,5,6,7,8,9,10])

data_scale = {}
data_stdscale = {}

feature_names = ["pfCand_pt", "pfCand_eta", "pfCand_phi", "pfCand_mass", "pfCand_numberOfHits",\
        "pfCand_dxy", "pfCand_dxy_error", "pfCand_dz", "pfCand_dz_error", "pfCand_track_chi2", \
        "pfCand_track_ndof","pfCand_caloFraction", "pfCand_hcalFraction", "pfCand_rawCaloFraction", "pfCand_rawHcalFraction",\
        "pfCand_rel_eta", "pfCand_rel_phi"]
mymax  = np.array([1000 , 3.5     ,  3.1416  , 0.1    , 55   , 5        , 10     , 100      , 10     , 1500 ,\
                    40   , 1, 1      , 1.5     ,0        ,1,1])
mymin  = np.array([0    , -3.5    , -3.1416  , -0.1   , 0    , -5       , -1     , -100     , -1     , 0    ,\
                    0    , 0, 0      , 0       ,1        ,-1,-1])
mymean = np.array([1.652,-0.002828, -0.006841, 0.09627, 6.288, -0.007103, 0.02706, -0.004195, 0.04633, 11.13,\
                    14.06, 0, 0.05801, 0.007524, 0.005249])
mystd  = np.array([4.407, 1.327   , 1.838    , 0.06446, 8.484, 1.873    , 0.8858 , 6.708    , 0.2561 , 58.9 ,\
                    6.76 ,0, 0.2338 , 0.08013  , 0.0674])
# Is it normal that caloFraction seems to be jsut 0 all the time?

for i in scale_list:
    print(feature_names[i])
    mymaxy = mymax[i]
    myminy = mymin[i]
    data_scale[feature_names[i]] = []
    data_scale[feature_names[i]].append({
        'max' : mymaxy,
        'min' : myminy,
    })

for i in stdscale_list:
    print(feature_names[i])
    mean  = mymean[i]
    std   = mystd[i]
    data_stdscale[feature_names[i]] = []
    data_stdscale[feature_names[i]].append({
        'std' : std,
        'mean': mean
    })

with open('mean_std.txt', 'w') as outfile:
    json.dump(data_stdscale, outfile) # writes the data object to the outfile file
with open('min_max.txt', 'w') as outfile:
    json.dump(data_scale, outfile) # writes the data object to the outfile file