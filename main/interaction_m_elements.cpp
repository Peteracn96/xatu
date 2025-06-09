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

    if (argc != 5){
		throw std::invalid_argument("Error: One input file is expected, nReciprocalVectors and ncell");
	}
    else if (argc < 5){
        throw std::invalid_argument("Error: At least one input file is required (system config.), Gcutoff, nReciprocalVectors and ncell");
    };
    
    int nbands = 1;
    int nrmbands = 0;
    arma::rowvec Q = {0., 0., 0.};
    arma::rowvec parameters = {1., 1., 10.};
    std::string modelfile = argv[1];    
    int nstates = 8;

    double Gcutoff = std::stod(argv[2]);
    int nReciprocalVectors = std::stoi(argv[3]);
    int ncell = std::stoi(argv[4]);

    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "|                                  Parameters                               |" << endl;
    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "#bands: " << nbands << endl;
    cout << "#removed bands: " << nrmbands << endl;
    cout << "System configuration file: " << modelfile << "\n" << endl;

    xatu::SystemConfiguration config(modelfile);
    
    auto start = high_resolution_clock::now();

    cout << "N. cells: " << ncell*ncell << endl;

    xatu::ExcitonTB bulkExciton(config, ncell, nbands, nrmbands, parameters, Q);

    arma::cout << "Orbitals: " << bulkExciton.system->orbitals << arma::endl;

    cout << "Valence bands:\n" << bulkExciton.valenceBands << endl;
    cout << "Conduction bands:\n" << bulkExciton.conductionBands << endl;
    cout << "Gauge used: " << bulkExciton.gauge << "\n" << endl;
    
    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "|                                Initialization                             |" << endl;
    cout << "+---------------------------------------------------------------------------+" << endl;
    bulkExciton.setPotential("keldysh");
    bulkExciton.brillouinZoneMesh(ncell);
    bulkExciton.initializeHamiltonian();

    bulkExciton.CompareInteractionMatrixElements(Gcutoff,nReciprocalVectors,"keldysh");

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    std::cout << "Elapsed time: " << duration.count()/1000.0 << " s" << std::endl;
    start = stop;

    return 0;
};