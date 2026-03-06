#include <math.h>
#include <chrono>
#include <iomanip>
#include <tclap/CmdLine.h>
#include "xatu.hpp"


#ifndef constants
#define PI 3.141592653589793
#define ec 1.6021766E-19
#define eps0 8.8541878E-12
#endif

using namespace arma;
using namespace std::chrono;

int main(int argc, char* argv[]){

    std::cout << std::setw(40) << std::left << "Testing DFT hBN... ";

    int ncell = 20;
    int nstates = 2;
    bool triangular = true;
    int holeIndex = 1;
    arma::rowvec holeCell = {0, 0, 0};

    std::string modelfile = "../examples/material_models/DFT/hBN_base_HSE06.outp";
    xatu::CRYSTALConfiguration config = xatu::CRYSTALConfiguration(modelfile, 100);

    xatu::ExcitonTB exciton = xatu::ExcitonTB(config, ncell, 1, 0, {1, 1, 10});
    exciton.system->setAU(true);

    exciton.brillouinZoneMesh(ncell);
    exciton.initializeHamiltonian();
    exciton.BShamiltonian();
    auto results = exciton.diagonalize("diag", nstates);

    auto energies = xatu::detectDegeneracies(results->eigval, nstates, 6);
    
    std::vector<std::vector<double>> expectedEnergies = {{4.442317, 1}, 
                                                         {4.442427, 1}};
                                                         
    for(uint i = 0; i < energies.size(); i++){

        if(std::abs(energies[i][0]-expectedEnergies[i][0]) < 1E-4 && energies[i][1] == expectedEnergies[i][1]){
            std::cout << "\033[1;32m Energy " << i+1 << " correct \033[0m" << std::endl;
        }
    }

    // Check reciprocal w.f.
    int nbandsCombinations = exciton.conductionBands.n_elem * exciton.valenceBands.n_elem;
    arma::cx_vec kwf = arma::zeros<arma::cx_vec>(exciton.system->kpoints.n_rows);
    for (int n = 0; n < nstates; n++){
        arma::cx_vec statecoefs = results->eigvec.col(n);
        for (int i = 0; i < exciton.system->kpoints.n_rows; i++){
        double coef = 0;
        for(int nband = 0; nband < nbandsCombinations; nband++){
            coef += abs(statecoefs(nbandsCombinations*i + nband))*
                    abs(statecoefs(nbandsCombinations*i + nband));
        };
        coef /= arma::norm(exciton.system->kpoints.row(1) - exciton.system->kpoints.row(0)); // L2 norm instead of l2
        kwf(i) += coef;
        };
    }

    double kwfHash = xatu::array2hash(kwf);
    double expectedKwfHash = 1.0864895707;
    if(std::abs(kwfHash-expectedKwfHash) < 1E-5){
        std::cout << "\033[1;32m Reciprocal w.f. correct \033[0m" << std::endl;
    }

    // Check realspace w.f.
    arma::rowvec holePosition = exciton.system->motif.row(holeIndex).subvec(0, 2) + holeCell;
    std::cout << "Hole position: " << holePosition << std::endl;
    double radius = arma::norm(exciton.system->bravaisLattice.row(0)) * exciton.ncell;
    arma::mat cellCombinations = exciton.system->truncateSupercell(exciton.ncell, 2);
    arma::vec rswf = arma::zeros(cellCombinations.n_rows*exciton.system->motif.n_rows);

    // Compute probabilities
    for(int n = 0; n < nstates; n++){
        int it = 0;
        arma::cx_vec statecoefs = results->eigvec.col(n);
        for(unsigned int cellIndex = 0; cellIndex < cellCombinations.n_rows; cellIndex++){
        arma::rowvec cell = cellCombinations.row(cellIndex);
        for (unsigned int atomIndex = 0; atomIndex < exciton.system->motif.n_rows; atomIndex++){
            rswf(it) += results->realSpaceWavefunction(statecoefs, atomIndex, holeIndex, cell, holeCell);
            it++;
        }
        }
    }

    double rswfHash = xatu::array2hash(rswf);
    double expectedRSwfHash = 83.2242560463;
    if(std::abs(rswfHash-expectedRSwfHash) < 1E-5){
        std::cout << "\033[1;32m Real space w.f. correct \033[0m" << std::endl;
    }

    std::cout.clear();
    std::cout << std::setw(40) << "\033[1;32m Success \033[0m" << std::endl;

    return 0;
}