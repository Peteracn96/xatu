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

    if (argc != 3){
		throw std::invalid_argument("Error: Two input files are expected");
	}
    else if (argc < 3){
        throw std::invalid_argument("Error: At least two input file are required (system config and screening config).");
    };
    
    int nbands = 1;
    int nrmbands = 0;
    arma::rowvec Q = {0., 0., 0.};
    arma::rowvec parameters = {1., 1., 10.};
    arma::rowvec q = {0.2, 0., 0}.;
    std::string modelfile = argv[1];
    std::string screeningfile = argv[2];    
    int nstates = 8;
    int Nk = 20;
    int nG = 25;
    double Gcutoff = 10.0;

    arma::vec Nk = arma::regspace(90, -10, 10);
    arma::vec Gcutoff_array = arma::regspace(0, 2., 10);

    bool writeEigvals = true;
    std::string filename = "inv_epsilon_vs_nGs_hBN_DFT_HSE06.dat";
    FILE* textfile_en = fopen(filename.c_str(), "a");

    

    xatu::SystemConfiguration config(modelfile); std::cout << "System configuration file parsed." << std::endl;
    xatu::ExcitonTB bulkExciton = xatu::ExcitonTB(config, ncell, nbands, nrmbands, parameters, Q, Gcutoff, nG); std::cout << "Exciton configuration initialized." << std::endl;
    xatu::ScreeningConfiguration screenconfig(screeningfile);

    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "|                                  Parameters                               |" << endl;
    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "N. cells: " << ncell*ncell << endl;
    cout << "#bands: " << nbands << endl;
    cout << "#removed bands: " << nrmbands << endl;
    cout << "System configuration file: " << modelfile << "\n" << endl;

    cout << "Valence bands:\n" << bulkExciton.valenceBands << endl;
    cout << "Conduction bands:\n" << bulkExciton.conductionBands << endl;
    cout << "Gauge used: " << bulkExciton.gauge << "\n" << endl;

    arma::cout << "Orbitals: " << bulkExciton.system->orbitals << arma::endl;
    bulkExciton.setMode("reciprocalspace");
    bulkExciton.setPotential("rpa");

    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "|                                Initialization                             |" << endl;
    cout << "+---------------------------------------------------------------------------+" << endl;

    bulkExciton.brillouinZoneMesh(ncell);
    bulkExciton.initializeHamiltonian();

    for(int i = 0; i < Gcutoff_array.n_elem; i++){

        auto start = high_resolution_clock::now();

        bulkExciton.setGcutoff(Gcutoff_array(i));

    

        if(writeEigvals){
            std::cout << "Writing eigvals to file: " << filename << std::endl;
            fprintf(textfile_en, "%d\t", ncell);
            results->writeEigenvalues(textfile_en, 8);
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        std::cout << "Elapsed time: " << duration.count()/1000.0 << " s" << std::endl;
        start = stop;

        // fprintf(textfile_en, "%10.6f\n", duration.count()/1000.0);
    }

    fclose(textfile_en);

    

    return 0;
};