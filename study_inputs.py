import ROOT as R
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train = R.RDataFrame('taus','/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root')#.Range(0,99000)

h = df_train.Histo1D(("", "; pt[GeV]; Number of entries", 1000, -100, 400),'pfCand_pt')
c = R.TCanvas('c', 'c', 500, 400)
c.Draw()
c.SetLogy()
h.Draw('same')
c.SaveAs('./Study_imputs_plots/pfCand_pt.pdf')

h = df_train.Histo1D(("", "; dxy ; Number of entries", 400, -200, 200),'pfCand_dxy')
c = R.TCanvas('c', 'c', 500, 400)
c.Draw()
c.SetLogy()
h.Draw('same')
c.SaveAs('./Study_imputs_plots/pfCand_dxy.pdf')

h = df_train.Histo1D(("", "; eta; Number of entries", 20, -10, 10),'pfCand_eta')
c = R.TCanvas('c', 'c', 500, 400)
c.Draw()
h.Draw('same')
c.SaveAs('./Study_imputs_plots/pfCand_eta.pdf')

h = df_train.Histo1D(("", "; fromPV; Number of entries", 50, -2, 5),'pfCand_fromPV')
c = R.TCanvas('c', 'c', 500, 400)
c.Draw()
h.Draw('same')
c.SaveAs('./Study_imputs_plots/pfCand_fromPV.pdf')


