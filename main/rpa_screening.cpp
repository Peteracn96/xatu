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

    xatu::SystemConfiguration model_config("../examples/material_models/wannier_models/MoS2_spin_wannier_07032024.model");

    xatu::ExcitonConfiguration exciton_config("../examples/excitonconfig/MoS2_test.txt");

    xatu::ScreeningConfiguration screening_config("../examples/screeningconfig/MoS2_wannier_screening.txt");

    xatu::ExcitonTB mos2_exciton(model_config, exciton_config,screening_config);

    mos2_exciton.brillouinZoneMesh(mos2_exciton.ncell);
    mos2_exciton.initializeHamiltonian();

    int NGs = mos2_exciton.nReciprocalVectors;

    std::cout << "Number of reciprocal vectors used in the calculation: " << NGs << std::endl;

    const int array_size = 10;
    int nGs_array[array_size] = {7,9,13,19,30,37,39,43,55,61};

    for (int gs = 0; gs < array_size; ++gs) {

	    int nGs = nGs_array[gs];

	    mos2_exciton.setReciprocalVectors(nGs);

        // Testing the code for RPA calculation
        mos2_exciton.compute_2D_RPAInvDielectricMatrix("MoS2_TB_q_points_test.dat");

        mos2_exciton.writeRPAPolarizabilityMatrix("MoS2_RPA_polarizability_" + std::to_string(nGs) + ".dat");

        mos2_exciton.writeRPAInverseDielectricMatrix("MoS2_TB_RPA_screening_" + std::to_string(nGs) + ".dat");
    }

    return 0;
}