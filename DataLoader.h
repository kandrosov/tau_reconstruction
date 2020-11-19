#include "TauTuple.h"

std::shared_ptr<TFile> OpenRootFile(const std::string& file_name){
    std::shared_ptr<TFile> file(TFile::Open(file_name.c_str(), "READ"));
    if(!file || file->IsZombie())
        throw std::runtime_error("File not opened.");
    return file;
}


struct Data {
    Data(size_t nx, size_t ny, size_t nz) : x(nx), y(ny), z(nz){ }
    std::vector<float> x, y, z;
};

// List of features:
enum class Feature {
    pfCand_pt                   = 0,
    pfCand_eta                  = 1,
    pfCand_phi                  = 2,
    pfCand_mass                 = 3,
    pfCand_pdgId                = 4,
    pfCand_charge               = 5,
    pfCand_pvAssociationQuality = 6,
    pfCand_fromPV               = 7,
    pfCand_puppiWeight          = 8, 
    pfCand_puppiWeightNoLep     = 9,
    pfCand_lostInnerHits        = 10,
    pfCand_numberOfPixelHits    = 11,
    pfCand_numberOfHits         = 12,
    pfCand_hasTrackDetails      = 13,
    pfCand_dxy                  = 14,
    pfCand_dxy_error            = 15,
    pfCand_dz                   = 16,
    pfCand_dz_error             = 17,
    pfCand_track_chi2           = 18,
    pfCand_track_ndof           = 19,
    pfCand_caloFraction         = 20,
    pfCand_hcalFraction         = 21,
    pfCand_rawCaloFraction      = 22,
    pfCand_rawHcalFraction      = 23,
};

string feature_names[24] = {"pfCand_pt", "pfCand_eta", "pfCand_phi", "pfCand_mass", "pfCand_pdgId", "pfCand_charge", 
    "pfCand_pvAssociationQuality", "pfCand_fromPV", "pfCand_puppiWeight", "pfCand_puppiWeightNoLep", "pfCand_lostInnerHits", 
    "pfCand_numberOfPixelHits", "pfCand_numberOfHits", "pfCand_hasTrackDetails", "pfCand_dxy", "pfCand_dxy_error", "pfCand_dz", 
    "pfCand_dz_error", "pfCand_track_chi2", "pfCand_track_ndof", "pfCand_caloFraction", "pfCand_hcalFraction", 
    "pfCand_rawCaloFraction", "pfCand_rawHcalFraction"};

struct DataLoader {

    //This is the constructor:
    DataLoader(std::string file_name, size_t _n_tau, Long64_t _start_dataset, Long64_t _end_dataset) :
        file(OpenRootFile(file_name)), 
        tuple(file.get(), true), 
        n_tau(_n_tau), 
        current_entry(_start_dataset), 
        start_dataset(_start_dataset), 
        end_dataset(_end_dataset){}
 
    std::shared_ptr<TFile> file;
    tau_tuple::TauTuple tuple; // tuple is the tree
    size_t n_tau; // number of events(=taus)
    Long64_t start_dataset;
    Long64_t end_dataset;
    Long64_t current_entry; // number of the current entry

    static const size_t n_pf    = 100; // number of pf candidates per event
    static const size_t n_fe    = 24;  // number of featurese per pf candidate
    static const size_t n_count = 2;   // chanrged and neutral particle count

    std::map<std::string, int> mapOfFeatures;
    std::map<std::string, int> MapCreation(){
        std::map<std::string, int> mapOfFeatures;
        for(size_t i = 0; i <= 23; ++i){
            mapOfFeatures[feature_names[i]] = i;
        }
        return mapOfFeatures;
    }

    bool HasNext() {     
        return (current_entry + n_tau) < end_dataset;
    }
    Data LoadNext(){
        Data data(n_tau * n_pf * n_fe, n_tau * n_count, n_tau); // Creates an empty data structure

        for(size_t tau_ind = 0; tau_ind < n_tau; ++tau_ind, ++current_entry) { 
            tuple.GetEntry(current_entry); // get the entry of the current event
            const tau_tuple::Tau& tau = tuple(); // tau is out tree

            data.z.at(tau_ind) = tau.tau_decayMode; 
            // Fill the labels:
            auto get_y = [&](size_t count_ind) -> float& {
                size_t index = GetIndex_y(tau_ind, count_ind);
                return data.y.at(index);
            };
            get_y(0) = Count_charged_hadrons_true(tau.lepton_gen_vis_pdg);
            get_y(1) = Count_neutral_hadrons_true(tau.lepton_gen_vis_pdg);

            for(size_t pf_ind = 0; pf_ind < n_pf; ++pf_ind) { 
                
                auto get_x = [&](Feature fe) -> float& {
                    size_t index = GetIndex_x(tau_ind, pf_ind, fe);
                    return data.x.at(index);
                };

                auto set_x_0 = [&](Feature fe) -> float& {
                    size_t index = GetIndex_x(tau_ind, pf_ind, fe);
                    return data.x.at(index) = 0;
                };
                
                size_t pf_size = tau.pfCand_pt.size(); 
                if(pf_ind < pf_size){
                    // Fill the data with the features:
                    // for(int i = 0; i <= 23; ++i){
                    //     std::cout << i << std::endl;
                    //     execute("get_x(Feature::"+feature_names[i]+") = tau."+feature_names[i]+".at(pf_ind)");
                    // }
                    get_x(Feature::pfCand_pt)                   = tau.pfCand_pt.at(pf_ind);
                    get_x(Feature::pfCand_eta)                  = tau.pfCand_eta.at(pf_ind);
                    get_x(Feature::pfCand_phi)                  = tau.pfCand_phi.at(pf_ind);
                    get_x(Feature::pfCand_mass)                 = tau.pfCand_mass.at(pf_ind);
                    get_x(Feature::pfCand_pdgId)                = tau.pfCand_pdgId.at(pf_ind);
                    get_x(Feature::pfCand_charge)               = tau.pfCand_charge.at(pf_ind);
                    get_x(Feature::pfCand_pvAssociationQuality) = tau.pfCand_pvAssociationQuality.at(pf_ind);
                    get_x(Feature::pfCand_fromPV)               = tau.pfCand_fromPV.at(pf_ind);
                    get_x(Feature::pfCand_puppiWeight)          = tau.pfCand_puppiWeight.at(pf_ind);
                    get_x(Feature::pfCand_puppiWeightNoLep)     = tau.pfCand_puppiWeightNoLep.at(pf_ind);
                    get_x(Feature::pfCand_lostInnerHits)        = tau.pfCand_lostInnerHits.at(pf_ind);
                    get_x(Feature::pfCand_numberOfPixelHits)    = tau.pfCand_numberOfPixelHits.at(pf_ind);
                    get_x(Feature::pfCand_numberOfHits)         = tau.pfCand_numberOfHits.at(pf_ind);
                    get_x(Feature::pfCand_hasTrackDetails)      = tau.pfCand_hasTrackDetails.at(pf_ind);
                    get_x(Feature::pfCand_dxy)                  = tau.pfCand_dxy.at(pf_ind);
                    get_x(Feature::pfCand_dz)                   = tau.pfCand_dz.at(pf_ind);
                    get_x(Feature::pfCand_caloFraction)         = tau.pfCand_caloFraction.at(pf_ind);
                    get_x(Feature::pfCand_hcalFraction)         = tau.pfCand_hcalFraction.at(pf_ind);
                    get_x(Feature::pfCand_rawCaloFraction)      = tau.pfCand_rawCaloFraction.at(pf_ind);
                    get_x(Feature::pfCand_rawHcalFraction)      = tau.pfCand_rawHcalFraction.at(pf_ind);

                    // Some features are invalid for the case where hasTrackDetails == 0:
                    if(tau.pfCand_hasTrackDetails.at(pf_ind) == 1){
                    get_x(Feature::pfCand_dxy_error)            = tau.pfCand_dxy_error.at(pf_ind);
                    get_x(Feature::pfCand_dz_error)             = tau.pfCand_dz_error.at(pf_ind);
                    get_x(Feature::pfCand_track_chi2)           = tau.pfCand_track_chi2.at(pf_ind);
                    get_x(Feature::pfCand_track_ndof)           = tau.pfCand_track_ndof.at(pf_ind);
                    }else{
                        set_x_0(Feature::pfCand_dxy_error);
                        set_x_0(Feature::pfCand_dz_error);
                        set_x_0(Feature::pfCand_track_chi2);
                        set_x_0(Feature::pfCand_track_ndof);
                    }
                } else{
                    // for(int i = 0; i <= 23; ++i){
                    //     set_x_0(Feature::feature_names[i]);
                    // }
                    set_x_0(Feature::pfCand_pt);
                    set_x_0(Feature::pfCand_eta);
                    set_x_0(Feature::pfCand_phi);
                    set_x_0(Feature::pfCand_mass);
                    set_x_0(Feature::pfCand_pdgId);
                    set_x_0(Feature::pfCand_charge);
                    set_x_0(Feature::pfCand_pvAssociationQuality);
                    set_x_0(Feature::pfCand_fromPV);
                    set_x_0(Feature::pfCand_puppiWeight);
                    set_x_0(Feature::pfCand_puppiWeightNoLep);
                    set_x_0(Feature::pfCand_lostInnerHits);
                    set_x_0(Feature::pfCand_numberOfPixelHits);
                    set_x_0(Feature::pfCand_numberOfHits);
                    set_x_0(Feature::pfCand_hasTrackDetails);
                    set_x_0(Feature::pfCand_dxy);
                    set_x_0(Feature::pfCand_dxy_error);
                    set_x_0(Feature::pfCand_dz);
                    set_x_0(Feature::pfCand_dz_error);
                    set_x_0(Feature::pfCand_track_chi2);
                    set_x_0(Feature::pfCand_track_ndof);
                    set_x_0(Feature::pfCand_caloFraction);
                    set_x_0(Feature::pfCand_hcalFraction);
                    set_x_0(Feature::pfCand_rawCaloFraction);
                    set_x_0(Feature::pfCand_rawHcalFraction);
                }
            }
            if(current_entry == end_dataset){
                tau_ind = n_tau;
            }
        }
        return data;
    }

    // Calculate the number of batches:
    size_t NumberOfBatches(){
        size_t n_entries = end_dataset-start_dataset;
        size_t n_batches = n_entries/n_tau;
        return n_batches;
    }

    // Resets the current entry to start_dataset so that we can loop on epochs:
    void Reset() {
        current_entry = start_dataset; 
    }

    // This function calculates the corresponding index in a 1D array: 
    // feed a = b.reshape(n_tau,n_pf,n_fe)
    size_t GetIndex_x(size_t tau_ind, size_t pf_ind, Feature fe) const {
        size_t fe_ind = static_cast<size_t>(fe);
        size_t part_pf_direction  = n_fe*pf_ind;
        size_t part_fe_direction  = fe_ind;
        size_t part_tau_direction = n_fe*n_pf*tau_ind;
        return part_fe_direction + part_pf_direction + part_tau_direction;
    }

    // This function calculates the corresponding index in a 1D array: 
    // feed a = b.reshape(n_tau,n_count)
    size_t GetIndex_y(size_t tau_ind, size_t count_ind) const {
        size_t part_count_direction = count_ind;
        size_t part_tau_direction   = n_count*tau_ind;
        return part_count_direction + part_tau_direction;
    }

    // Function that counts the true values for the number of charged hadrons for the event:
    size_t Count_charged_hadrons_true(std::vector<int> lepton_gen_vis_pdg){
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
    size_t Count_neutral_hadrons_true(std::vector<int> lepton_gen_vis_pdg){
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

};