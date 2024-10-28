#include <armadillo>
#include <xatu.hpp>
#include <iostream>
#include <set>

using namespace std::chrono;


int main(){

    auto start = high_resolution_clock::now();
    arma::rowvec q = {1.2, 0.3, 0};

    int nstates = 8;
    int decimals = 6;

    xatu::SystemConfiguration model_config("../examples/material_models/MoS2.model");

    xatu::ExcitonConfiguration exciton_config("../examples/excitonconfig/MoS2_test.txt");

    xatu::ScreeningConfiguration screening_config("../examples/screeningconfig/MoS2_TB_screening.txt");

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

    arma::mat Rdifferences(nRdif,5,arma::fill::zeros);

    int index_row = 0;
    for (int i = 0; i < nRvectors; i++)
    {
        arma::rowvec Ri = LatticeVectors.row(i);

        for (int j = 0; j < nRvectors; j++)
        {
            arma::rowvec Rj = LatticeVectors.row(j);

            Rdifferences.row(index_row).subvec(0,2) = Ri - Rj;
            Rdifferences(index_row,3) = i;
            Rdifferences(index_row,4) = j;
            ++index_row;
        }
    }

    std::cout << "nRdif = " << nRdif << std::endl;

    int index_array[nRdif]={0};

    for (arma::uword i = 0; i < nRdif; ++i){
        index_array[i] = i;
    }

    int index = 0;
    for (int i = 0; i < nRdif; ++i){
        arma::rowvec Rdif_aux = Rdifferences.row(i).subvec(0,2);
        for (int j = 0; j < nRdif; ++j) {
            arma::rowvec Rdif_aux2 = Rdifferences.row(j).subvec(0,2);
            if (arma::norm(Rdif_aux - Rdif_aux2) < 1E-7){
                index_array[index] = j;
                break;
            }
        }
        index++;
    }

    std::cout << "List of the non-equivalent vectors:\n";

    for (int i = 0; i < nRdif; ++i){
        arma::rowvec Rdif_aux = Rdifferences.row(i).subvec(0,2);
        std::cout << "index = " << index_array[i]  << ", R2-R1(" << i << ") = " << Rdif_aux  << std::endl;
    }

    std::set<int> indexes_set = std::set<int>(index_array, index_array + nRdif);

    int n_non_equivalent_vectors = indexes_set.size();

    int indexes_array[n_non_equivalent_vectors];

    int i_aux = 0;
    for (int const& index : indexes_set)
    {
        indexes_array[i_aux] = index;
        i_aux++;
    }

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

    // Generates all non equivalent to compute T

    int n_positive_vectors = (n_non_equivalent_vectors - 1)/2 + 1; // Needs only to compute the T blocks for the upper diagonal part. +1->is the origin
    int n_non_equivalent_combinations = n_positive_vectors*NAtoms*NAtoms;

    arma::imat non_equivalent_combinations(n_non_equivalent_combinations,4,arma::fill::zeros);

    arma::mat T_aux(n_positive_vectors*NAtoms,NAtoms,arma::fill::zeros);
    
    i_aux = 0;
    for (int index = 0; index < n_positive_vectors; ++index){
        for (int t_i = 0; t_i < NAtoms; ++t_i){
            for (int t_j = 0; t_j < NAtoms; ++t_j){
                non_equivalent_combinations(i_aux,0) = indexes_array[index]; // index of the row in the matrix storing all R differences
                non_equivalent_combinations(i_aux,1) = t_i;
                non_equivalent_combinations(i_aux,2) = t_j;
                non_equivalent_combinations(i_aux,3) = index; // index in the array storing the indexes of the non equivalent R differences
                ++i_aux;
            }
        }
    }

    // Computes non equivalent matrix elements of T

    std::cout << "Computing all the elements... " << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < n_non_equivalent_combinations; i++)
    {
        int index = non_equivalent_combinations(i,3);
        int R_dif_index = non_equivalent_combinations(i,0);
        int t_i_index = non_equivalent_combinations(i,1);
        int t_j_index = non_equivalent_combinations(i,2);

        arma::rowvec R_dif = Rdifferences.row(R_dif_index).subvec(0,2);
        arma::rowvec R_origin(3,arma::fill::zeros);

        T_aux(index*NAtoms + t_i_index,t_j_index) = mos2_exciton.computesinglePolarizability(R_dif, R_origin, t_i_index, t_j_index);

        //std::cout << "i = " << i << ", R_dif = (" << R_dif(0) << "," << R_dif(1) << "), t_i = " << t_i_index << ", t_j = " << t_j_index  << std::endl;
        //std::cout << i << ", " << std::flush;
    }
    
    // Prints the elements
    for (arma::uword const& i : arma::regspace(0,9))
    {
        T_aux.submat(i*NAtoms, 0, i*NAtoms + NAtoms - 1, NAtoms - 1).print(std::to_string(i)+":");
    }
    //T_aux.print("T_aux:");
    
    
    // Builds the big T matrix

    i_aux = 0;

    for (int R_i = 0; R_i < nRvectors; R_i++){
        
        for (int R_j = R_i; R_j < nRvectors; R_j++){

            auto ptr = std::find(indexes_array, indexes_array + nRvectors*nRvectors, index_array[R_i*nRvectors + R_j]);
            int found_index = ptr - indexes_array;
            arma::mat T_aux_mat = T_aux.submat(found_index, 0, found_index + NAtoms - 1, NAtoms - 1);
            T.submat(R_i*NAtoms, R_j*NAtoms, R_i*NAtoms + NAtoms - 1, R_j*NAtoms + NAtoms - 1) = T_aux_mat;
            T.submat(R_j*NAtoms, R_i*NAtoms, R_j*NAtoms + NAtoms - 1, R_i*NAtoms + NAtoms - 1) = arma::trans(T_aux_mat);
        }
    }
    
    // Prints the T matrix

    T.print("T:");

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Done in " << duration.count()/1000.0 << " s." << std::endl;

    return 0;
}