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

    //auto start = high_resolution_clock::now();

    xatu::SystemConfiguration model_config("../examples/material_models/wannier_models/MoS2_spin_wannier_07032024_tb.model");

    xatu::ExcitonConfiguration exciton_config("../examples/excitonconfig/MoS2_test.txt");

    xatu::ScreeningConfiguration screening_config("../examples/screeningconfig/MoS2_wannier_screening.txt");

    xatu::ExcitonTB hBN_exciton(model_config, exciton_config,screening_config);

    hBN_exciton.brillouinZoneMesh(hBN_exciton.ncell);
    hBN_exciton.initializeHamiltonian();

    hBN_exciton.writeBZtofile();

    int NGs = hBN_exciton.nReciprocalVectors;

    std::cout << "Number of reciprocal vectors used in the calculation: " << NGs << std::endl;

    // const int array_size = 11;
    // int nGs_array[array_size] = {1,8,9,13,19,30,37,39,43,55,61};

    // for (int gs = 0; gs < array_size; ++gs) {

	//     int nGs = nGs_array[gs];

	//     hBN_exciton.setReciprocalVectors(nGs);

    //     // Testing the code for RPA calculation
    //     hBN_exciton.compute_2D_RPAInvDielectricMatrix("MoS2_TB_q_points_test.dat");

    //     hBN_exciton.writeRPAPolarizabilityMatrix("MoS2_RPA_polarizability_" + std::to_string(nGs) + ".dat");

    //     hBN_exciton.writeRPAInverseDielectricMatrix("MoS2_TB_RPA_screening_" + std::to_string(nGs) + ".dat");
    // }

    //hBN_exciton.compute_2D_DielectricMatrix_Opt("MoS2_TB_q_points_test.dat"); not optimal at the end

    hBN_exciton.compute_2D_DielectricMatrix("qy_points_MoS2_spin_wannier_semicore.dat");

    //hBN_exciton.invertDielectricMatrix();

    //hBN_exciton.writePolarizabilityMatrix("MoS2_polarizability.dat");

    //hBN_exciton.writeInverseDielectricMatrix("MoS2_TB_screening.dat");

    hBN_exciton.writeDielectricMatrix("MoS2_spin_wannier_semicore_qy.dat");

    // Testing the code for RPA calculation
    // hBN_exciton.compute_2D_RPAInvDielectricMatrix("MoS2_TB_q_points_test.dat");

    // hBN_exciton.writeRPAPolarizabilityMatrix("MoS2_RPA_polarizability.dat");

    // hBN_exciton.writeRPAInverseDielectricMatrix("MoS2_TB_RPA_screening.dat");

    return 0;
}