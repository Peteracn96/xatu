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
    arma::rowvec parameters = {1., 1., 10.};
    std::string modelfile = argv[1];    
    int nstates = 8;
    int nG = 25;
    double Gcutoff = 10.0;

    arma::vec ncell_array = arma::regspace(40, -10, 10);

    bool writeEigvals = true;
    std::string filename = "eigval_convergence_hbn_reciprocal_nG25_noReg.out";
    FILE* textfile_en = fopen(filename.c_str(), "a");

    for(int i = 0; i < ncell_array.n_elem; i++){

        auto start = high_resolution_clock::now();

        int ncell = ncell_array(i);

        cout << "+---------------------------------------------------------------------------+" << endl;
        cout << "|                                  Parameters                               |" << endl;
        cout << "+---------------------------------------------------------------------------+" << endl;
        cout << "N. cells: " << ncell*ncell << endl;
        cout << "#bands: " << nbands << endl;
        cout << "#removed bands: " << nrmbands << endl;
        cout << "System configuration file: " << modelfile << "\n" << endl;

        xatu::SystemConfiguration config(modelfile);
        xatu::ExcitonTB bulkExciton = xatu::ExcitonTB(config, ncell, nbands, nrmbands, parameters, Q, Gcutoff, nG);

        arma::cout << "Orbitals: " << bulkExciton.system->orbitals << arma::endl;
        bulkExciton.setMode("reciprocalspace");
        bulkExciton.setReciprocalVectors(nG);
        bulkExciton.setPotential("keldysh");

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

        if(writeEigvals){
            std::cout << "Writing eigvals to file: " << filename << std::endl;
            fprintf(textfile_en, "%d\t", ncell);
            results->writeEigenvalues(textfile_en, 8);
        }

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        std::cout << "Elapsed time: " << duration.count()/1000.0 << " s" << std::endl;
        start = stop;

        fprintf(textfile_en, "%10.6f\n", duration.count()/1000.0);
    }

    fclose(textfile_en);

    

    return 0;
};