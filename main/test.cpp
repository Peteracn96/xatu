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

    std::string modelfile = "../examples/material_models/DFT/hBN_base_HSE06.outp";
    std::unique_ptr<xatu::SystemConfiguration> systemConfig = std::unique_ptr<xatu::CRYSTALConfiguration>(new xatu::CRYSTALConfiguration(modelfile, 100));
    std::string dielectric_matrix_path = "../data/hBN_DFT_HSE06_Q2D_BZ_N20_Gc_5_1_invepsilon.dat";

    int ncell = 20;
    double Gcutoff = 5.1;
    double d = 3.33;
    double percentage = 0.5;
    int nstates = 2;
    uint nvbands = 6;
    uint ncbands = 63;
    uint ncell_aux = 5;
    bool spinfull = false;
    bool isotropic = true;
    int holeIndex = 1;
    arma::rowvec holeCell = {0, 0, 0};

    xatu::ExcitonTB exciton = xatu::ExcitonTB(*systemConfig, ncell, 1, 0, {1., 1., 10.}, {0., 0., 0.}, ncell_aux, nvbands, ncbands, Gcutoff, Gcutoff, d, spinfull, isotropic);
    
    exciton.system->setAU(true);
    exciton.setPercentage(percentage);

    exciton.brillouinZoneMesh(ncell);
    exciton.readInverseDielectricMatrix(dielectric_matrix_path);
    exciton.initializeHamiltonian();

    // std::complex<double> epsilon = exciton.computesingleDielectricFunctionMatrixElement();

    exciton.BShamiltonian();

    auto results = exciton.diagonalize("diag", nstates);

    auto energies = xatu::detectDegeneracies(results->eigval, nstates, 6);

    cout << "+---------------------------------------------------------------------------+" << endl;
    cout << "|                                    Results                                |" << endl;
    cout << "+---------------------------------------------------------------------------+" << endl;

    xatu::printEnergies(results, nstates, 10);
    
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
    /*double expectedKwfHash = 1.0864895707;
    if(std::abs(kwfHash-expectedKwfHash) < 1E-5){
        std::cout << "\033[1;32m Reciprocal w.f. correct \033[0m" << std::endl;
    }*/
    std::cout << std::setw(40) << "kwfHash: " << kwfHash << std::endl;
    printf("|%.15lf|\n", kwfHash);

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
    /*double expectedRSwfHash = 83.2242560463;
    if(std::abs(rswfHash-expectedRSwfHash) < 1E-5){
        std::cout << "\033[1;32m Real space w.f. correct \033[0m" << std::endl;
    }*/
    std::cout << std::setw(40) << "rswfHash: " << rswfHash << std::endl; 
    printf("|%.15lf|\n", rswfHash);
    std::cout.clear();
    std::cout << std::setw(40) << "\033[1;32m Success \033[0m" << std::endl;

    return 0;
}