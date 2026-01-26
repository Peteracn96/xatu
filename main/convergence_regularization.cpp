#include <iostream>
#include <fstream>
#include <armadillo>
#include <complex>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <chrono>

#include <xatu.hpp>

#ifndef constants
#define PI 3.141592653589793
#define ec 1.6021766E-19
#define eps0 8.8541878E-12
#endif

using namespace arma;
using namespace std::chrono;

int main(int argc, char* argv[]){

    // Parse console stdin
    int n_args = 5;

    if (argc != n_args){
		throw std::invalid_argument("Error: Two input files are expected");
	}
    else if (argc < n_args){
        throw std::invalid_argument("Error: At least two input file are required (system config, exciton config, screening config and screening data file).");
    };

    std::string modelfile = argv[1];
    std::string excitonfile = argv[2];
    std::string screeningfile = argv[3];
    std::string inv_epsilon_file = argv[4];

    int nstates = 8;
    int decimals = 6;

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

    screeningConfig.reset(new xatu::ScreeningConfiguration(screeningfile));
    excitonConfig.reset(new xatu::ExcitonConfiguration(excitonfile));


    xatu::ExcitonTB exciton = xatu::ExcitonTB(*systemConfig, *excitonConfig, *screeningConfig);
    // xatu::ExcitonTB exciton = xatu::ExcitonTB(*systemConfig, *excitonConfig);

    exciton.setMode(excitonConfig->excitonInfo.mode);
    
    if (modelfile.find(".outp") != std::string::npos){
        exciton.system->setAU(true); // if input model is CRYSTAL
    }

    arma::rowvec Q = {0., 0., 0.};
    arma::rowvec parameters = {1., 1., 5.172};
    std::string modelfile = argv[1];    
    int nstates = 8;
    double Gc_exciton = 20;
    double Gcutoff = 20.0;

    // int ncell = 60;

    arma::vec percentage_array = {1.1, 1.2, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55};

    bool writeEigvals = true;
    std::string filename = "eigval_convergence_hBN_HSE06_Gc_37_0_with_reg.out";
    FILE* textfile_en = fopen(filename.c_str(), "a");

    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "|                                  Parameters                               |" << endl;
    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "N. cells: " << exciton.ncell*exciton.ncell << endl;

    // cout << "#bands: " << nbands << endl;
    cout << "System configuration file: " << modelfile << "\n" << endl;

    // xatu::SystemConfiguration config(modelfile);

    // std::cout << "System configuration file parsed." << std::endl;
    
    // xatu::ExcitonTB exciton = xatu::ExcitonTB(config, ncell, nbands, nrmbands, parameters, Q, Gcutoff, nG); 

    exciton.setMode("reciprocalspace");
    exciton.setPotential("rpa");

    cout << "Valence bands:\n" << exciton.valenceBands << endl;
    cout << "Conduction bands:\n" << exciton.conductionBands << endl;
    cout << "Gauge used: " << exciton.gauge << "\n" << endl;
    
    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "|                                Initialization                             |" << endl;
    cout << "+---------------------------------------------------------------------------+" << endl;

    exciton.brillouinZoneMesh(exciton.ncell);
    exciton.initializeHamiltonian();

    exciton.printInformation();

    exciton.readInverseDielectricMatrix(inv_epsilon_file);
    
    for(int i = 0; i < percentage_array.n_elem; i++){

        auto start = high_resolution_clock::now();

        cout << "Percentage: " << percentage_array(i) << endl;

        exciton.setPercentage(percentage_array(i));
        
        exciton.BShamiltonian();
        auto results = exciton.diagonalize("diag", nstates);

        cout << "+---------------------------------------------------------------------------+" << endl;
        cout << "|                                    Results                                |" << endl;
        cout << "+---------------------------------------------------------------------------+" << endl;

        printEnergies(results, nstates, 6);

        cout << "+---------------------------------------------------------------------------+" << endl;
        cout << "|                                    Output                                 |" << endl;
        cout << "+---------------------------------------------------------------------------+" << endl;

        fprintf(textfile_en, "%10.6f ", percentage_array(i));

        if(writeEigvals){
            std::cout << "Writing eigvals to file: " << filename << std::endl;
            fprintf(textfile_en, "%d\t", exciton.ncell);
            results->writeEigenvalues(textfile_en, 8);
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        std::cout << "Elapsed time: " << duration.count()/1000.0 << " s" << std::endl;
        start = stop;

        
    }

    fclose(textfile_en);

    return 0;
};