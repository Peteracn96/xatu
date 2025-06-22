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

    if (argc != 4){
		throw std::invalid_argument("Error: Two input files are expected");
	}
    else if (argc < 4){
        throw std::invalid_argument("Error: At least two input file are required (system config, exciton config and screening config).");
    };
    
    int nbands = 1;
    int nrmbands = 0;
    arma::rowvec Q = {0., 0., 0.};
    arma::rowvec q = {0.2, 0., 0.};
    arma::rowvec parameters = {1., 1., 10.};
    arma::mat q_points_list = {{0.2, 0., 0.}};
    
    std::string modelfile = argv[1];
    std::string excitonfile = argv[2];
    std::string screeningfile = argv[3];
    
    int nstates = 8;
    int Nk = 20;
    int nG = 25;
    double Gcutoff = 10.0;
    int ncell = 20;
    arma::vec Gcutoff_array = arma::regspace(3., 3., 54);

    // FILE* textfile_en = fopen(filename.c_str(), "a");

    // xatu::ExcitonConfiguration systemConfig;

    std::unique_ptr<xatu::SystemConfiguration> systemConfig;
    std::unique_ptr<xatu::ExcitonConfiguration> excitonConfig;
    std::unique_ptr<xatu::ScreeningConfiguration> screeningConfig;

    systemConfig.reset(new xatu::CRYSTALConfiguration(modelfile, 100));
    //systemConfig.reset(new xatu::SystemConfiguration(modelfile));
    screeningConfig.reset(new xatu::ScreeningConfiguration(screeningfile));
    excitonConfig.reset(new xatu::ExcitonConfiguration(excitonfile));

    xatu::ExcitonTB bulkExciton = xatu::ExcitonTB(*systemConfig, *excitonConfig, *screeningConfig);
    bulkExciton.setMode(excitonConfig->excitonInfo.mode);
    bulkExciton.system->setAU(true);

    std::cout << "System configuration file parsed." << std::endl;

    //xatu::ExcitonTB bulkExciton = xatu::ExcitonTB(*systemConfig, ncell, nbands, nrmbands, parameters, Q, Gcutoff, nG);

    std::cout << "Exciton configuration initialized." << std::endl;

    std::string filename =excitonConfig->excitonInfo.label +  "inv_epsilon_vs_nGs_symmetrized";

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

    bulkExciton.setq_points_list(q_points_list);

    uint nGs_aux = 0;

    arma::ivec nGs_array(25,arma::fill::zeros);
    uint j = 0; // Number of points in the plot will be j-1

    /*for (int i = 0; i < Gcutoff_array.n_elem; i++)
    {


        double gc = Gcutoff_array(i);

        bulkExciton.setGcutoff(gc);

        // If the new cutoff generates the same number of Gs as the previous iteration, increment i
        if (nGs_aux == bulkExciton.getNGs())
        {
            ++i;
            continue;
        }

        nGs_aux = bulkExciton.getNGs();
        nGs_array(j) = nGs_aux;
        ++j;
    }

    std::cout << "Number of Gs for each Gcutoff: " << nGs_array(j - 1) << std::endl;*/

    
    for(int i = 0; i < Gcutoff_array.n_elem; i++){

        auto start = high_resolution_clock::now();

        double gc = Gcutoff_array(i);

        std::cout << "Gcutoff: " << gc << std::endl;
        std::cout << "nGs_aux: " << nGs_aux << std::endl;
        bulkExciton.setGcutoff(gc);

        // If the new cutoff generates the same number of Gs as the previous iteration, increment i
        if (nGs_aux == bulkExciton.getNGs()){
            ++i;
            continue;
        }

        nGs_aux = bulkExciton.getNGs();

        std::cout << "nGs: " << bulkExciton.getNGs() << std::endl;

        bulkExciton.compute_2D_DielectricMatrix_at_q(q,0);

        bulkExciton.invertDielectricMatrix();

        bulkExciton.writeInverseDielectricMatrix(filename + "_" + std::to_string(nGs_aux) + ".dat");

        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        std::cout << "Elapsed time: " << duration.count()/1000.0 << " s" << std::endl;
        start = stop;

        ++j;

        std::cout << "Computation " << (i + 1)*100 / Gcutoff_array.n_elem << "%% complete." << std::endl;
    }

    return 0;
};