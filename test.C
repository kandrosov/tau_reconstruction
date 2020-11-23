#include "TROOT.h"
#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"

int test(){
    TFile *f = new TFile("/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root");
    TTree *taus = (TTree*)f->Get("taus");

    // TFile *f = 0;
    // f = new TFile("/data/store/reco_skim_v1/tau_DYJetsToLL_M-50.root","READ")
    // TTree *t = nullptr;
    // f.GetObject("taus", t);
    // TBranch *bran = tau->GetBranch("pfCand_pt");

    // taus->Draw("pfCand_pt");
    // taus->Print();

    // TCanvas *c1 = new TCanvas("c1","c1",10,10,1000,750);
    // c1->Divide(2,2);
    // c1->cd(1); 
    // gPad->SetGrid();
    // taus->Draw("pfCand_pt");
    // taus->Show();

    const char * feature_names[2] = {"pfCand_pt>>h0", "pfCand_eta>>h1", "pfCand_phi", "pfCand_mass", "pfCand_pdgId", "pfCand_charge", 
    "pfCand_pvAssociationQuality", "pfCand_fromPV", "pfCand_puppiWeight", "pfCand_puppiWeightNoLep", "pfCand_lostInnerHits", 
    "pfCand_numberOfPixelHits", "pfCand_numberOfHits", "pfCand_hasTrackDetails", "pfCand_dxy", "pfCand_dxy_error", "pfCand_dz", 
    "pfCand_dz_error", "pfCand_track_chi2", "pfCand_track_ndof", "pfCand_caloFraction", "pfCand_hcalFraction", 
    "pfCand_rawCaloFraction", "pfCand_rawHcalFraction"};


    TCanvas *c0 = new TCanvas;
    taus->Draw("pfCand_pt>>h0");

    TCanvas *c1 = new TCanvas;
    taus->Draw("pfCand_eta>>h1");


    gSystem->ProcessEvents();

    TImage *img = TImage::Create();
    img->FromPad(c0);
    img->FromPad(c1);
    img->WriteImage("canvas.png");

    h0->GetXaxis()->GetXmax()
    h0->GetXaxis()->GetXmin()
    h1->GetXaxis()->GetXmax()
    h1->GetXaxis()->GetXmin()

    delete h0;
    delete c0;
    delete h1;
    delete c1;
    delete img;

   return 0;
}