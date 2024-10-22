#include <armadillo>
#include <xatu.hpp>
#include <iostream>
#include <set>

int main(){

    arma::rowvec q = {1.2, 0.3, 0};

    int nstates = 8;
    int decimals = 6;

    xatu::SystemConfiguration model_config("MoS2.model");

    xatu::ExcitonConfiguration exciton_config("MoS2_test.txt");

    xatu::ScreeningConfiguration screening_config("MoS2_TB_screening.txt");

    xatu::ExcitonTB mos2_exciton(model_config, exciton_config,screening_config);

    mos2_exciton.brillouinZoneMesh(mos2_exciton.ncell);
    mos2_exciton.initializeHamiltonian();


    mos2_exciton.setVectors(0,0);
    //mos2_exciton.computesingleDielectricFunction();

    std::cout << std::left << std::setw(30) << "Dielectric constant of embedding medium: " << mos2_exciton.eps_m << std::endl;
    std::cout << std::left << std::setw(30) << "Dielectric constant of substrate: " << mos2_exciton.eps_s << std::endl;

    //FILE* energies_file = fopen("mos2_exciton_nGs.txt", "w");

    int NGs = mos2_exciton.getNGs();

    int nGsarray[] = {7,13,19,31,37,43};

    // for (int ngs : nGsarray){
    //     mos2_exciton.setReciprocalVectors(ngs);
    //     mos2_exciton.BShamiltonian();

    //     auto results = mos2_exciton.diagonalize("diag", nstates);

    //     fprintf(energies_file, "%d\t ", ngs);

    //     for(int i = 0; i < nstates; i++){
    //         fprintf(energies_file, "%f\t ", results->eigval(i));
    //     }
    //     fprintf(energies_file, "\n");
    // }

    // double radius = arma::norm(mos2_exciton.system->bravaisLattice.row(0)) * cutoff_;
    // arma::mat lattice_vectors = mos2_exciton.system->truncateSupercell(ncell, radius);
    
    arma::mat LatticeVectors = mos2_exciton.trunLattice_;
    int nRvectors = LatticeVectors.n_rows;

    //Prints values of the polarizability in the lattice
    for (int i = 0; i < nRvectors; i++)
    {
        arma::rowvec Raux = LatticeVectors.row(i);

        std::cout << "R(" << i << ") = " << Raux << std::endl;
    }

    int nRdif = nRvectors*nRvectors;

    arma::mat Rdifferences(nRdif,3,arma::fill::zeros);

    int index_row = 0;
    for (int i = 0; i < nRvectors; i++)
    {
        arma::rowvec Ri = LatticeVectors.row(i);

        for (int j = 0; j < nRvectors; j++)
        {
            arma::rowvec Rj = LatticeVectors.row(j);

            Rdifferences.row(index_row) = Ri - Rj;
            ++index_row;
        }
    }

    arma::ivec CountRdifs(nRdif,arma::fill::zeros);

    int count = 0;
    int non_equivalent = 0;

    std::cout << "nRdif = " << nRdif << std::endl;

    int index_array[nRdif]={0};

    int index = 0;
    for (int i = 0; i < nRdif; ++i){
        arma::rowvec Rdif_aux = Rdifferences.row(i);
        for (int j = 0; j < nRdif; ++j) {
            arma::rowvec Rdif_aux2 = Rdifferences.row(j);
            if (arma::norm(Rdif_aux - Rdif_aux2) < 1E-7){
                index_array[index] = j;
                break;
            }
        }
        index++;
    }

    for (int i = 0; i < nRdif; ++ i){
        arma::rowvec Rdif_aux = Rdifferences.row(i);
        //std::cout << "R2-R1(" << i << ") = " << Rdif_aux << std::endl;
    }

    std::cout << "List of the indexes of the non-equivalent vectors:\n";

    for (int i = 0; i < nRdif; ++i){
        arma::rowvec Rdif_aux = Rdifferences.row(i);
        std::cout << "index = " << index_array[i]  << ", R2-R1(" << i << ") = " << Rdif_aux  << std::endl;
    }


    // for (int i = 0; i < nRdif; ++ i){
    //     std::cout << "Vector dif. nummer " << i << " was found " << CountRdifs(i) << " times\n";
    // }

    std::set<int> indexes_set = std::set<int>( index_array, index_array + nRdif );

    std::cout << "The indices are:" <<  std::endl;

    for (int const& index : indexes_set)
    {
        std::cout << index << ", ";
    }

    std::cout << "\nTotal number of non equivalent vectors is = " << indexes_set.size() << " vectors" <<  std::endl;

    std::cout << "Number of k points in BZ mesh = " << mos2_exciton.system->kpoints.n_rows << std::endl;

    return 0;
}