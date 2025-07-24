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
    arma::rowvec parameters = {1., 1., 5.172};
    std::string modelfile = argv[1];    
    int nstates = 8;
    double Gc_exciton = 20;
    double Gcutoff = 20.0;

    int ncell = 10;

    arma::vec percentage_array = {0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0};

    bool writeEigvals = true;
    std::string filename = "eigval_convergence_hBN_HSE06_Gc_20_0_with_reg.out";
    FILE* textfile_en = fopen(filename.c_str(), "a");

    for(int i = 0; i < percentage_array.n_elem; i++){

        auto start = high_resolution_clock::now();

        cout << "+---------------------------------------------------------------------------+" << endl;
        cout << "|                                  Parameters                               |" << endl;
        cout << "+---------------------------------------------------------------------------+" << endl;
        cout << "N. cells: " << ncell*ncell << endl;
        cout << "#bands: " << nbands << endl;
        cout << "#removed bands: " << nrmbands << endl;
        cout << "System configuration file: " << modelfile << "\n" << endl;
        cout << "Percentage: " << percentage_array(i) << endl;

        // xatu::SystemConfiguration config(modelfile);

        std::unique_ptr<xatu::SystemConfiguration> systemConfig;
        systemConfig.reset(new xatu::CRYSTALConfiguration(modelfile, 100));

        std::cout << "System configuration file parsed." << std::endl;
        
        xatu::ExcitonTB bulkExciton = xatu::ExcitonTB(*systemConfig, ncell, nbands, nrmbands, parameters, Q, Gcutoff, Gc_exciton); 
        // xatu::ExcitonTB bulkExciton = xatu::ExcitonTB(config, ncell, nbands, nrmbands, parameters, Q, Gcutoff, nG); 

        bulkExciton.system->setAU(true); // Comment if input model is not CRYSTAL

        arma::cout << "Orbitals: " << bulkExciton.system->orbitals << arma::endl;

        bulkExciton.setMode("reciprocalspace");
        bulkExciton.setPotential("keldysh");

        bulkExciton.setPercentage(percentage_array(i));

        cout << "Valence bands:\n" << bulkExciton.valenceBands << endl;
        cout << "Conduction bands:\n" << bulkExciton.conductionBands << endl;
        cout << "Gauge used: " << bulkExciton.gauge << "\n" << endl;
        
        cout << "+---------------------------------------------------------------------------+" << endl;
        cout << "|                                Initialization                             |" << endl;
        cout << "+---------------------------------------------------------------------------+" << endl;

        bulkExciton.brillouinZoneMesh(ncell);
        bulkExciton.initializeHamiltonian();
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