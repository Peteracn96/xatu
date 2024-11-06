#include <armadillo>
#include <xatu.hpp>
#include <iostream>
#include <set>

using namespace std::chrono;


double my_coulomb(double r) {
    
    double a = 0.000001;
    double v_c_regularization =ec/(4E-10*PI*eps0*a);
    
    if (r < 1E-7){
        //return 0;
        return v_c_regularization;
    }

    return ec/(4E-10*PI*eps0*r);    
    //return 1/r;
}

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

    //Prints vectors of the the lattice
    for (int i = 0; i < nRvectors; i++)
    {
        arma::rowvec Raux = LatticeVectors.row(i);

        std::cout << "R(" << i << ") = " << Raux << std::endl;
    }

    int nRdif = (nRvectors*nRvectors-1)/2 + nRvectors;

    arma::mat Rdifferences(nRdif,5,arma::fill::zeros);

    int index_row = 0;
    for (int i = 0; i < nRvectors; i++)
    {
        arma::rowvec Ri = LatticeVectors.row(i);

        for (int j = i; j < nRvectors; j++)
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

    //std::cout << "List of the non-equivalent vectors:\n";

    //for (int i = 0; i < nRdif; ++i){
    //    arma::rowvec Rdif_aux = Rdifferences.row(i).subvec(0,2);
        //std::cout << "index = " << index_array[i]  << ", R2-R1(" << i << ") = " << Rdif_aux  << std::endl;
    //}

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

    for (int const& index : indexes_array)
    {
        std::cout << index << ", ";
    }

    std::cout << "\nTotal number of non equivalent vectors is = " << n_non_equivalent_vectors << " vectors" <<  std::endl;

    std::cout << "Number of k points in BZ mesh = " << mos2_exciton.system->kpoints.n_rows << std::endl;

    // Initiates the polarizability and computes it at the non-equivalent sites

    int NAtoms = mos2_exciton.system->natoms;

    int n_rows = nRvectors*NAtoms;

    arma::cx_mat T(n_rows,n_rows,arma::fill::zeros);

    // Generates all non equivalent to compute T

    int n_positive_vectors = n_non_equivalent_vectors;//(n_non_equivalent_vectors - 1)/2 + 1; Needs only to compute the T blocks for the upper diagonal part. +1->is the origin
    int n_non_equivalent_combinations = n_positive_vectors*NAtoms*NAtoms;

    arma::imat non_equivalent_combinations(n_non_equivalent_combinations,4,arma::fill::zeros);

    arma::cx_mat T_aux(n_positive_vectors*NAtoms,NAtoms,arma::fill::zeros);
    
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

        T_aux(index*NAtoms + t_i_index,t_j_index) = -mos2_exciton.realPolarizabilityMatrixElement(R_dif, R_origin, t_i_index, t_j_index);

        //std::cout << "i = " << i << ", R_dif = (" << R_dif(0) << "," << R_dif(1) << "), t_i = " << t_i_index << ", t_j = " << t_j_index  << std::endl;
        //std::cout << i << ", " << std::flush;
    }
    
    // Prints the elements
    // for (arma::uword const& i : arma::regspace(0,9))
    // {
    //     T_aux.submat(i*NAtoms, 0, i*NAtoms + NAtoms - 1, NAtoms - 1).print(std::to_string(i)+":");
    // }
    //T_aux.print("T_aux:");
    
    
    // Builds the big T matrix
    std::cout << "T_aux number of rows: " << T_aux.n_rows << std::endl;
    i_aux = 0;

    for (int R_i = 0; R_i < nRvectors; R_i++){
        
        for (int R_j = R_i; R_j < nRvectors; R_j++){

            // auto ptr = std::find(indexes_array, indexes_array + nRvectors*nRvectors, index_array[R_i*nRvectors + R_j]);
            // int found_index = ptr - indexes_array;
            int found_index = 0;
            arma::rowvec R_dif_aux = LatticeVectors.row(R_i) - LatticeVectors.row(R_j);

            for (int j = 0; j < nRdif; ++j) {
                arma::rowvec Rdif_aux2 = Rdifferences.row(j).subvec(0,2);
                if (arma::norm(R_dif_aux - Rdif_aux2) < 1E-7){
                    found_index = j;
                    break;
                }
            }
            
            auto ptr = std::find(indexes_array, indexes_array + n_non_equivalent_vectors, found_index);
            int found_index_2 = ptr - indexes_array;
            std::cout << found_index_2 << ", " << std::flush;
            arma::cx_mat T_aux_mat = T_aux.submat(found_index_2*NAtoms, 0, found_index_2*NAtoms + NAtoms - 1, NAtoms - 1);
            T.submat(R_i*NAtoms, R_j*NAtoms, R_i*NAtoms + NAtoms - 1, R_j*NAtoms + NAtoms - 1) = T_aux_mat;
            T.submat(R_j*NAtoms, R_i*NAtoms, R_j*NAtoms + NAtoms - 1, R_i*NAtoms + NAtoms - 1) = arma::trans(T_aux_mat);
        }
    }
    
    // Prints the T matrix

    //T.print("T:");

    // Now is the inversion of Dyson's equation

    int n_positions = nRvectors*NAtoms - 1; // Minus one position, as we throw away the terms of the form V(t_j,t_j)/W(t_j,t_j) 
    arma::cx_mat V(n_positions, NAtoms, arma::fill::zeros);
    arma::cx_mat W(n_positions, NAtoms, arma::fill::zeros); 
    arma::cx_cube epsilon(n_positions,n_positions,NAtoms);

    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        epsilon.slice(t_j) = arma::cx_mat(n_positions,n_positions,arma::fill::zeros);
    }

    arma::ucube combinations(n_positions,2,NAtoms);

    // Generates all the combinations, for each t_j

    int origin_index = (nRvectors - 1)/2;
    int origin_index_aux = origin_index*NAtoms;

    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        int index_aux = 0;

        for (arma::uword R_i = 0; R_i < nRvectors; ++R_i){

            if (R_i == origin_index){

                for (arma::uword t_i = 0; t_i < NAtoms; ++t_i){

                    if (t_i != t_j){
                        combinations.slice(t_j).row(index_aux)(0) = R_i;
                        combinations.slice(t_j).row(index_aux)(1) = t_i;
                        ++index_aux;
                    }
                }
            } else {

                for (arma::uword t_i = 0; t_i < NAtoms; ++t_i){
                    combinations.slice(t_j).row(index_aux)(0) = R_i;
                    combinations.slice(t_j).row(index_aux)(1) = t_i;
                    ++index_aux;
                }
            }
            
        }
    }

    // Prints the combinations
    // for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
    //     combinations.slice(t_j).print("t_j = " + std::to_string(t_j));
    // }

    //Prints combinations of selected t_j
    //combinations.slice(0).print("Combinations for t_j=0");

    // Computes the bare Coulomb potential matrix

    arma::mat motif = mos2_exciton.system->motif;

    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
        arma::rowvec t_j_vector = motif.row(t_j).subvec(0,2);

        for (arma::uword index = 0; index < n_positions; ++index){

            int R_i = combinations.slice(t_j).row(index)(0);
            int t_i = combinations.slice(t_j).row(index)(1);

            arma::rowvec R = LatticeVectors.row(R_i);
            arma::rowvec t = motif.row(t_i).subvec(0,2);
            V.col(t_j)(index) = my_coulomb(arma::norm(R + t - t_j_vector));
        }
    }

    // Computes the "epsilon" matrices

    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
        arma::rowvec t_j_vector = motif.row(t_j).subvec(0,2);

        for (arma::uword index = 0; index < n_positions; ++index){

            arma::rowvec R = LatticeVectors.row(combinations.slice(t_j).row(index)(0));
            arma::rowvec t_i = motif.row(combinations.slice(t_j).row(index)(1)).subvec(0,2);

            for (arma::uword index2 = 0; index2 < n_positions; ++index2){

                //Lambda function to compute the sum
                auto sum_func = [index2,&R,t_j,&t_i,nRvectors,NAtoms,&combinations,&T,&LatticeVectors,&motif,&mos2_exciton]() -> std::complex<double> {
                    std::complex<double> sum = 0;

                    int Rprime_index = combinations.slice(t_j).row(index2)(0);
                    int tprime_index = combinations.slice(t_j).row(index2)(1);
                    int pos_prime_index = Rprime_index*NAtoms + tprime_index;
                    arma::rowvec Rprime_vector = LatticeVectors.row(Rprime_index);
                    arma::rowvec tprime_vector = motif.row(tprime_index).subvec(0,2);

                    for (arma::uword R2 = 0; R2 < nRvectors; ++R2){

                        arma::rowvec R2_vector = LatticeVectors.row(R2);

                        for (arma::uword i2 = 0; i2 < NAtoms; ++i2){

                            arma::rowvec t2 = motif.row(i2).subvec(0,2);

                            double norm = arma::norm(R + t_i - (R2_vector + t2));
                            
                            if (norm > 1E-7){
                                //sum += my_coulomb(norm)*mos2_exciton.realPolarizabilityMatrixElement(R2_vector, Rprime_vector, i2, tprime_index);
                                sum += my_coulomb(norm)*T(R2*NAtoms + i2,pos_prime_index);
                            }
                        }
                    }

                    return sum;
                };
                double kronecker_delta = index == index2 ? 1.0:0.0;
                epsilon.slice(t_j)(index,index2) = kronecker_delta - sum_func();
            }
        }
    }

    // Prints the epsilon matrices
    // for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
    //     epsilon.slice(t_j).print(std::to_string(t_j) + ":");
    // }
    NAtoms = 1;
    // Solves for W
    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        W.col(t_j) = arma::solve(epsilon.slice(t_j), V.col(t_j));
    }

    // Prints the V columns
    // for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
    //     V.col(t_j).print("V(.,t_" + std::to_string(t_j) + "):");
    // }
    
    // Prints the W columns
    // for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
    //     W.col(t_j).print("W(.,t_" + std::to_string(t_j) + "):");
    // }

    // Prints the V and W columns side by side
    NAtoms = 1;
    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
        for (int pos = 0; pos < n_positions; ++pos){

            int R_index = combinations.slice(t_j).row(pos)(0);
            int t_index = combinations.slice(t_j).row(pos)(1);
            
            arma::rowvec R = LatticeVectors.row(R_index);
            arma::rowvec t_i = motif.row(t_index).subvec(0,2);

            //std::cout << "R(" << R_index << ") = ("  << std::setw(5) << R(0)  <<"," << R(1) << "," << R(2) <<")," << "t(" << t_index << ") = ("  << t_i(0) << "," << t_i(1) << "," << t_i(2) << ")," << std::right << "V = " << real(V.col(t_j)(pos)) << ", W = " <<  real(W.col(t_j)(pos)) << ";\n";
            std::cout << "R(" << R_index << "), t(" << t_index <<  std::setw(2) << "), V = " << real(V.col(t_j)(pos)) << ", W = " <<  real(W.col(t_j)(pos)) << ";\n";

        }
        
    }


    // Prints the W columns in mathematica format
    /*for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
        for (arma::uword i = 0; i < W.col(t_j).n_rows; ++i){
            arma::rowvec R = LatticeVectors.row(combinations.row(i)(0));
            arma::rowvec t_i = motif.row(combinations.row(i)(1)).subvec(0,2);
            
            std::cout << "{{" << R(0) + t_i(0) <<"," << R(1) + t_i(1) << "," << R(2) + t_i(2) <<"}," <<  real(W.col(t_j)(i)) << "},";
        }
    }lolo*/

    // FILE* textfile = fopen("test_W_screening.dat", "w");

    // if (textfile == NULL){
    //     std::cout << "File for inverse of the dielectric matrix failed to open. Exiting" << std::endl;
    //     exit(0);
    // }

    // std::cout << "Writing screened potential fo file: " << textfile << std::endl;
    // for(arma::uword t_j = 0; t_j < NAtoms; t_j++){
    //     for (arma::uword pos_index = 0; pos_index < n_positions; pos_index++){
    //         arma::rowvec R = LatticeVectors.row(combinations.slice(t_j).row(pos_index)(0));
    //         arma::rowvec t = motif.row(combinations.slice(t_j).row(pos_index)(1));
    //         fprintf(textfile, "%11.7lf %11.7lf %11.7lf %11.7lf %11.7lf %11.7lf %11.7lf\n", R(0), R(1), R(2), t(0), t(1), t(2), W.col(t_j)(pos_index));
    //     }

    //     fprintf(textfile, "\n");
    // }

    // fclose(textfile);

    /****************Now is the inversion of Dyson's equation including singularities*****************/

    /*int npositions = nRvectors*NAtoms; // Minus one position, as we throw away the terms of the form V(t_j,t_j)/W(t_j,t_j) 
    arma::cx_mat V_2(npositions, NAtoms, arma::fill::zeros);
    arma::cx_mat W_2(npositions, NAtoms, arma::fill::zeros); 
    arma::cx_mat epsilon_2(npositions,npositions,arma::fill::zeros);

    arma::umat combinations_2(npositions,2);

    // Generates all the combinations, for each t_j
    
    int index_aux = 0;
    for (arma::uword R_i = 0; R_i < nRvectors; ++R_i){
        for (arma::uword t_i = 0; t_i < NAtoms; ++t_i){
            combinations_2.row(index_aux)(0) = R_i;
            combinations_2.row(index_aux)(1) = t_i;
            ++index_aux;
        }
    }

    // Prints the combinations
    // for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
    //     combinations.slice(t_j).print("t_j = " + std::to_string(t_j));
    // }

    // Computes the bare Coulomb potential matrix

    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
        arma::rowvec t_j_vector = motif.row(t_j).subvec(0,2);

        for (arma::uword index = 0; index < npositions; ++index){

            int R_i = combinations_2.row(index)(0);
            int t_i = combinations_2.row(index)(1);

            arma::rowvec R = LatticeVectors.row(R_i);
            arma::rowvec t = motif.row(t_i).subvec(0,2);

            double norm = arma::norm(R + t - t_j_vector);

            V_2.col(t_j)(index) = my_coulomb(norm);
        }
    }

    // Computes the "epsilon" matrices


    for (arma::uword index = 0; index < npositions; ++index){

        arma::rowvec R = LatticeVectors.row(combinations_2.row(index)(0));
        arma::rowvec t_i = motif.row(combinations_2.row(index)(1)).subvec(0,2);

        for (arma::uword index2 = 0; index2 < npositions; ++index2){

            //Lambda function to compute the sum
            auto sum_func = [index2,&R,&t_i,nRvectors,NAtoms,&combinations_2,&T,&LatticeVectors,&motif]() -> std::complex<double> {
                std::complex<double> sum = 0;

                int Rprime_index = combinations_2.row(index2)(0);
                int tprime_index = combinations_2.row(index2)(1);
                int pos_prime_index = Rprime_index*NAtoms + tprime_index;

                for (arma::uword R2 = 0; R2 < nRvectors; ++R2){

                    arma::rowvec R2_vector = LatticeVectors.row(R2);

                    for (arma::uword i2 = 0; i2 < NAtoms; ++i2){

                        arma::rowvec t2 = motif.row(i2).subvec(0,2);

                        double norm = arma::norm(R + t_i - (R2 + t2));

                        double v_c = my_coulomb(norm);
                        
                        sum += v_c*T(R2*NAtoms + i2,pos_prime_index);
                    }
                }

                return sum;
            };
            double kronecker_delta = index == index2 ? 1.0:0.0;
            epsilon_2(index,index2) = kronecker_delta - sum_func();
        }
    }

    // Prints the epsilon matrices
    // for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
    //     epsilon.slice(t_j).print(std::to_string(t_j) + ":");
    // }

    // Inverts "epsilon"
    arma::cx_mat Inv_epsilon = epsilon_2.i();

    epsilon_2.print("epsilon_2:");

    Inv_epsilon.print("Inv_epsilon:");

    // Solves for W
    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        W_2.col(t_j) = Inv_epsilon*V_2.col(t_j);
    }

    // Prints the V columns
    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
        V_2.col(t_j).print("V_2(.,t_" + std::to_string(t_j) + "):");
    }

    // Prints the W columns
    for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
        W_2.col(t_j).print("W_2(.,t_" + std::to_string(t_j) + "):");
    }

    // Prints the W columns in mathematica format
    // for (arma::uword t_j = 0; t_j < NAtoms; ++t_j){
        
    //     for (arma::uword i = 0; i < W_2.col(t_j).n_rows; ++i){
    //         arma::rowvec R = LatticeVectors.row(combinations_2.row(i)(0));
    //         arma::rowvec t_i = motif.row(combinations_2.row(i)(1)).subvec(0,2);
            
    //         std::cout << "{{" << R(0) + t_i(0) <<"," << R(1) + t_i(1) << "," << R(2) + t_i(2) <<"}," <<  real(W_2.col(t_j)(i)) << "},";
    //     }
    // }

    FILE* textfile_2 = fopen("test_sing_W_screening.dat", "w");

    if (textfile_2 == NULL){
        std::cout << "File for inverse of the dielectric matrix failed to open. Exiting" << std::endl;
        exit(0);
    }

    std::cout << "Writing inverse of dielectric matrix fo file: " << textfile_2 << std::endl;
    for(arma::uword t_j = 0; t_j < NAtoms; t_j++){
        for (arma::uword pos_index = 0; pos_index < npositions; pos_index++){
            arma::rowvec R = LatticeVectors.row(combinations_2.row(pos_index)(0));
            arma::rowvec t = motif.row(combinations_2.row(pos_index)(1));
            fprintf(textfile_2, "%11.7lf %11.7lf %11.7lf %11.7lf %11.7lf %11.7lf %11.7lf\n", R(0), R(1), R(2), t(0), t(1), t(2), W_2.col(t_j)(pos_index));
        }

        fprintf(textfile_2, "\n");
    }

    fclose(textfile_2);*/

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "\nDone in " << duration.count()/1000.0 << " s." << std::endl;

    return 0;
}