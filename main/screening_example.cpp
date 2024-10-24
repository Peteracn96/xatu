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

    std::set<int> indexes_set = std::set<int>(index_array, index_array + nRdif);

    int n_non_equivalent_vectors = indexes_set.size();

    std::cout << "The indices are:" <<  std::endl;

    for (int const& index : indexes_set)
    {
        std::cout << index << ", ";
    }

    std::cout << "\nTotal number of non equivalent vectors is = " << n_non_equivalent_vectors << " vectors" <<  std::endl;

    std::cout << "Number of k points in BZ mesh = " << mos2_exciton.system->kpoints.n_rows << std::endl;

    // Initiates the polarizability and computes it at the non-equivalent sites

    int NAtoms = mos2_exciton.system->natoms;

    int n_rows = nRvectors*NAtoms;

    arma::mat T(n_rows,n_rows,arma::fill::zeros);

    // Generates combinations (all of them and non equivalent ones) to compute T

    int n_all_combinations = pow(nRvectors*NAtoms,2);
    arma::imat all_combinations(n_all_combinations,4,arma::fill::zeros);
    int i_aux = 0;
    for (int R_i = 0; R_i < nRvectors; R_i++){
        
        for (int R_j = 0; R_j < nRvectors; R_j++){
            
            for (int t_i = 0; t_i < NAtoms; ++t_i){
                
                for (int t_j = 0; t_j < NAtoms; ++t_j){
                    all_combinations(i_aux,0) = R_i;
                    all_combinations(i_aux,1) = t_i;
                    all_combinations(i_aux,2) = R_j;
                    all_combinations(i_aux,3) = t_j;
                    ++i_aux;
                }
            }
        }
    }

    int n_non_equivalent_combinations = n_non_equivalent_vectors*NAtoms*NAtoms;

    arma::imat non_equivalent_combinations(n_non_equivalent_combinations,3,arma::fill::zeros);

    arma::vec T_aux(n_non_equivalent_combinations,arma::fill::zeros);
    
    i_aux = 0;
    for (int const& index : indexes_set){
        for (int t_i = 0; t_i < NAtoms; ++t_i){
            for (int t_j = 0; t_j < NAtoms; ++t_j){
                non_equivalent_combinations(i_aux,0) = index;
                non_equivalent_combinations(i_aux,1) = t_i;
                non_equivalent_combinations(i_aux,2) = t_j;
                ++i_aux;
            }
        }
    }

    // Computes non equivalent matrix elements of T

    for (int i = 0; i < n_non_equivalent_combinations; i++)
    {
        int R_dif_index = non_equivalent_combinations(i,0);
        int t_i_index = non_equivalent_combinations(i,1);
        int t_j_index = non_equivalent_combinations(i,2);

        arma::rowvec R_dif = Rdifferences.row(R_dif_index);
        arma::rowvec R_origin(3,arma::fill::zeros);

        T_aux(i) = mos2_exciton.computesinglePolarizability(R_dif, R_origin, t_i_index, t_j_index);
        std::cout << "i = " << i << ", R_dif = (" << R_dif(0) << "," << R_dif(1) << "), t_i = " << t_i_index << ", t_j = " << t_j_index << std::endl;
    }

    // Prints the elements
    for (int i = 0; i < n_non_equivalent_combinations; i++)
    {
        std::cout << "T(" << i << ") = " << T_aux(i) << std::endl;
    }
    
    // Builds the big T matrix

    for (int i = 0; i < n_non_equivalent_combinations; i++)
    {
        int R_dif_index = non_equivalent_combinations(i,0);

        int R_i = Rdifferences.row(R_dif_index)(3);
        int R_j = Rdifferences.row(R_dif_index)(4);

        int t_i_index = non_equivalent_combinations(i,1);
        int t_j_index = non_equivalent_combinations(i,2);

        T(R_i + t_i_index, R_j + t_j_index) = T_aux(i);
    }
    
    // Prints the T matrix

    T.print("T:");

    return 0;
}