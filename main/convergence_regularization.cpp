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

    if (argc != 2){
		throw std::invalid_argument("Error: One input file is expected");
	}
    else if (argc < 2){
        throw std::invalid_argument("Error: At least one input file is required (system config.)");
    };
    
    int nbands = 1;
    int nrmbands = 0;
    arma::rowvec Q = {0., 0., 0.};
    arma::rowvec parameters = {1., 1., 36.2};
    std::string modelfile = argv[1];    
    int nstates = 8;
    double Gc_exciton = 18.0;
    double Gcutoff = 18.0;

    int ncell = 60;

    arma::vec percentage_array = {1.35,1.4,1.45,1.5,1.55,1.6,1.65,1.7,1.75,1.8,1.9,2.0};

    bool writeEigvals = true;
    std::string filename = "eigval_convergence_MoS2_with_reg_extra.out";
    FILE* textfile_en = fopen(filename.c_str(), "a");

    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "|                                  Parameters                               |" << endl;
    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "N. cells: " << ncell*ncell << endl;
    cout << "#bands: " << nbands << endl;
    cout << "System configuration file: " << modelfile << "\n" << endl;

    // xatu::SystemConfiguration config(modelfile);

    std::unique_ptr<xatu::SystemConfiguration> systemConfig;
    systemConfig.reset(new xatu::CRYSTALConfiguration(modelfile, 100));

    std::cout << "System configuration file parsed." << std::endl;
    
    xatu::ExcitonTB bulkExciton = xatu::ExcitonTB(*systemConfig, ncell, nbands, nrmbands, parameters, Q, Gcutoff, Gc_exciton); 
    // xatu::ExcitonTB bulkExciton = xatu::ExcitonTB(config, ncell, nbands, nrmbands, parameters, Q, Gcutoff, nG); 

    bulkExciton.system->setAU(true); // Comment if input model is not CRYSTAL

    bulkExciton.setMode("reciprocalspace");
    bulkExciton.setPotential("keldysh");

    bulkExciton.printInformation();

    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "|                                Initialization                             |" << endl;
    cout << "+---------------------------------------------------------------------------+" << endl;

    bulkExciton.brillouinZoneMesh(ncell);
    bulkExciton.initializeHamiltonian();

    for(int i = 0; i < percentage_array.n_elem; i++){

        auto start = high_resolution_clock::now();

        cout << "Percentage: " << percentage_array(i) << endl;

        bulkExciton.setPercentage(percentage_array(i));
        
        bulkExciton.BShamiltonian();
        auto results = bulkExciton.diagonalize("diag", nstates);

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
            fprintf(textfile_en, "%d\t", ncell);
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