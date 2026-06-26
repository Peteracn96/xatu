#include <armadillo>
#include <xatu.hpp>
#include <iostream>
#include <set>

using namespace std::chrono;

// run command: ./dynamic_screening ../examples/material_models/DFT/hBN_base_HSE06.outp ../examples/screeningconfig/hBN_DFT_screening.txt <name_of_q_points_file>.dat wi wf Nws T FermiEnergy eta <output_file>.dat

int main(int argc, char* argv[]){

    // Parse console stdin

    if (argc != 11){
		throw std::invalid_argument("Error: Eleven input arguments are expected");
	}
    else if (argc < 11){
        throw std::invalid_argument("Error: At least eleven input arguments are required (system config, exciton config, screening config, q points file and output file).");
    };

    std::string modelfile = argv[1];    
    std::string screeningfile = argv[2];
    std::string q_points_file = argv[3];
    std::string wi_s = argv[4];
    std::string wf_s = argv[5];
    std::string Nws_s = argv[6];
    std::string T_s = argv[7];
    std::string FermiEnergy_s = argv[8];
    std::string eta_s = argv[9];
    std::string output_file = argv[10];

    double wi = std::stod(wi_s);
    double wf = std::stod(wf_s);
    int Nws = std::stoi(Nws_s);
    double T = std::stod(T_s);
    double FermiEnergy = std::stod(FermiEnergy_s);
    double eta = std::stod(eta_s);

    arma::mat q_points;
    // Reads q points from file
    std::ifstream inputfile(q_points_file);
    if (!inputfile) {
        std::cout << "File for q points failed to open or does not exist. Exiting" << std::endl;
        exit(1);
    }

    std::string line;
    double qx, qy, qz;

    try{
        while(std::getline(inputfile, line)){
            std::istringstream iss(line);
            iss >> qx >> qy >> qz;
            arma::rowvec qpoint{qx, qy, qz};
            q_points.insert_rows(q_points.n_rows,qpoint);
        }
        inputfile.close();
    }
    catch(const std::exception& e){
        std::cerr << e.what() << std::endl;
    }

    std::unique_ptr<xatu::SystemConfiguration> systemConfig;
    std::unique_ptr<xatu::ExcitonConfiguration> excitonConfig;
    std::unique_ptr<xatu::ScreeningConfiguration> screeningConfig;

    if (modelfile.find(".outp") != std::string::npos){
        systemConfig.reset(new xatu::CRYSTALConfiguration(modelfile, 100));
    }
    else if (modelfile.find(".model") != std::string::npos){
        systemConfig.reset(new xatu::SystemConfiguration(modelfile));
    } else {
        throw std::invalid_argument("Error: Unsupported system configuration file format. Use .outp or .model files.");
    }

    uint ncell = 20;
    uint ncell_aux = 700;
    uint nvbands = 1;
    uint ncbands = 1;
    double Gcutoff = 0.1;
    double d = 3.22;
    bool spinfull = false;
    bool isotropic = true;

    screeningConfig.reset(new xatu::ScreeningConfiguration(screeningfile));
    xatu::ExcitonTB exciton = xatu::ExcitonTB(*systemConfig, ncell, 1, 0, {1., 1., 0.5}, {0., 0., 0.}, ncell_aux, nvbands, ncbands, Gcutoff, Gcutoff, d, spinfull, isotropic);
    
    exciton.setMode("reciprocalspace");
    exciton.setTemperature(T);
    exciton.setFermiEnergy(FermiEnergy);
    exciton.setBroadening(eta);

    if (modelfile.find(".outp") != std::string::npos){
        exciton.system->setAU(true); // if input model is CRYSTAL
    }
    
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                                  Parameters                               |" << std::endl;
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    
    std::cout << "System configuration file: " << modelfile << "\n" << std::endl;
    std::cout << "q points file file:        " << q_points_file << "\n" << std::endl;

    exciton.brillouinZoneMesh(exciton.ncell);
    exciton.initializeHamiltonian();    

    // exciton.compute_2D_DielectricMatrix(wi, wf, Nws, q_points, output_file);
    exciton.compute_2D_Polarizability(wi, wf, Nws, 0, 0, q_points, output_file);

    return 0;
}