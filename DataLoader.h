#include "TauTuple.h"


std::shared_ptr<TFile> OpenRootFile(const std::string& file_name)
{
    std::shared_ptr<TFile> file(TFile::Open(file_name.c_str(), "READ"));
    if(!file || file->IsZombie())
        throw std::runtime_error("File not opened.");
    return file;
}


struct Data {
    Data(size_t nx, size_t ny) : x(nx), y(ny) { }
    std::vector<float> x, y;
};

// List of features:
enum class Feature {
    pfCand_pt = 0,
    pfCand_eta = 1,
    pfCand_phi = 2,
};


struct DataLoader {
    //This is the constructor:
    DataLoader(std::string file_name, size_t _n_tau) :
    // DataLoader(std::string file_name, size_t _n_tau, size_t start_dataset, size_t end_dataset) :
        file(OpenRootFile(file_name)), tuple(file.get(), true), n_tau(_n_tau), current_entry(0) { }
    
    std::shared_ptr<TFile> file;
    tau_tuple::TauTuple tuple; // tuple is the tree
    size_t n_tau; // number of events/taus
    Long64_t current_entry; // number of the current entry

    static const size_t n_pf  = 5; // number of pf candidates per event
    static const size_t n_fe  = 30; // number of featurese per pf candidate
    static const size_t n_count = 2; // chanrged and neutral particle count

    bool HasNext() {        
        return current_entry + n_tau < tuple.GetEntries();
    }
    Data LoadNext(){
        
        //Create an empty data structure:
        Data data(n_tau * n_pf * n_fe, n_tau * n_count);

        for(size_t tau_ind = 0; tau_ind < n_tau; ++tau_ind, ++current_entry) { 
            tuple.GetEntry(current_entry); // get the entry of the current event
            const tau_tuple::Tau& tau = tuple();
            for(size_t pf_ind = 0; pf_ind < n_pf; ++pf_ind) {

                auto get_x = [&](Feature fe) -> float& {
                    size_t index = GetIndex_x(tau_ind, pf_ind, fe);
                    return data.x.at(index);
                };

                auto get_y = [&](size_t count_ind) -> float& {
                    size_t index = GetIndex_y(tau_ind, count_ind);
                    return data.y.at(index);
                };

                // std::cout << tau_ind << " " << pf_ind << " " << tau.pfCand_pt.at(pf_ind) << " "
                //           << tau.pfCand_eta.at(pf_ind) <<  " " << tau.pfCand_phi.at(pf_ind) << std::endl;

                size_t pf_size = tau.pfCand_pt.size(); 
                if(pf_ind < pf_size){
                    // Fill the data with the features
                    get_x(Feature::pfCand_pt)  = tau.pfCand_pt.at(pf_ind);
                    get_x(Feature::pfCand_eta) = tau.pfCand_eta.at(pf_ind);
                    get_x(Feature::pfCand_phi) = tau.pfCand_phi.at(pf_ind);

                    get_y(0) = Count_charged_hadrons_true(tau.lepton_gen_vis_pdg);
                    get_y(1) = Count_neutral_hadrons_true(tau.lepton_gen_vis_pdg);
                } else{
                    get_x(Feature::pfCand_pt)  = 0;
                    get_x(Feature::pfCand_eta) = 0;
                    get_x(Feature::pfCand_phi) = 0;
                }
            }
        }
        return data;
    }

    // Calculate the number of batches:
    size_t NumberOfBatches(){
        size_t n_entries = tuple.GetEntries();
        size_t n_batches = n_entries/n_tau;
        return n_batches;
    }

    // Resets the current entry to zero so that we can loop on epochs:
    void Reset() {
        current_entry = 0; 
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

    size_t GetIndex_y(size_t tau_ind, size_t count_ind) const {
        size_t part_count_direction = count_ind;
        size_t part_tau_direction   = n_count*tau_ind;
        return part_count_direction + part_tau_direction;
    }

    // Functions that count the true values for the number of charged and neutral hadrons for the event:
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

    size_t Count_neutral_hadrons_true(std::vector<int> lepton_gen_vis_pdg){
        size_t cnt_neutral_hadrons = 0;
        for(int pdg_id : lepton_gen_vis_pdg) {
            pdg_id = std::abs(pdg_id);
            if(pdg_id == 22 || pdg_id == 310 || pdg_id == 130){
                ++cnt_neutral_hadrons;
            }
        }
        return cnt_neutral_hadrons;
    }

};