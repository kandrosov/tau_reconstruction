#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"

TFile f("/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root");
TTree *t = nullptr;
f.GetObject("taus", t);

Float_t mean = tree->GetMean();