#include <armadillo>
#include <xatu.hpp>

using namespace xatu;

int main(){

    int nbands = 2;
    int nrmbands = 0;
    int ncell = 30;
    arma::rowvec parameters = {1., 4., 13.55};
    int nstates = 16;
    arma::rowvec Q = arma::rowvec{0., 0., 0.};

    auto config = SystemConfiguration("./models/MoS2.model");
    auto exciton = Exciton(config, ncell, nbands, nrmbands, parameters, Q);

    FILE* energies_file = fopen("mos2_exciton_bands.txt", "w");

    double reciprocal_distance = arma::norm(exciton.reciprocalLattice.row(0))/2.;

    for(int n = -20; n <= 20; n++){

        Q(0) = 1.325597163545547 + reciprocal_distance*n/200;

        exciton.setExchange(true);
        exciton.setQ(Q);

        exciton.brillouinZoneMesh(ncell);

        exciton.initializeHamiltonian();
        exciton.BShamiltonian();
        auto results = exciton.diagonalize("diag", nstates);

        fprintf(energies_file, "%f\t", Q(0));
        for(int i = 0; i < nstates; i++){
            fprintf(energies_file, "%f\t", results.eigval(i));
        }
        fprintf(energies_file, "\n");
    }

    return 0;
};