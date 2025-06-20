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
    
    /* xatu::SystemConfiguration model_config("../examples/material_models/wannier_models/MoS2_spin_wannier_07032024_tb.model");
    xatu::ExcitonConfiguration exciton_config("../examples/excitonconfig/MoS2_test.txt");

    xatu::ScreeningConfiguration screening_config("../examples/screeningconfig/MoS2_Wannier_screening.txt");
    
    xatu::ExcitonTB hBN_exciton(model_config, exciton_config,screening_config);
    
    */
    
    std::unique_ptr<xatu::SystemConfiguration> systemConfig;
    std::unique_ptr<xatu::ExcitonConfiguration> excitonConfig;
    std::unique_ptr<xatu::ScreeningConfiguration> screeningConfig;

    systemConfig.reset(new xatu::CRYSTALConfiguration("../examples/material_models/DFT/hBN_base_HSE06.outp", 169));
    screeningConfig.reset(new xatu::ScreeningConfiguration("../examples/screeningconfig/hBN_DFT_screening.txt"));
    excitonConfig.reset(new xatu::ExcitonConfiguration("../examples/excitonconfig/hBN_test.txt"));


    xatu::ExcitonTB hBN_exciton = xatu::ExcitonTB(*systemConfig, *excitonConfig, *screeningConfig);
    hBN_exciton.setMode(excitonConfig->excitonInfo.mode);
    hBN_exciton.system->setAU(true);


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

    std::string q_points_file = "hBN_DFT_HSE06_q_point.dat";
    hBN_exciton.compute_2D_DielectricMatrix(q_points_file);

    hBN_exciton.invertDielectricMatrix();

    hBN_exciton.writePolarizabilityMatrix("../hBN_DFT_HSE06_q_polarizability.dat");

    hBN_exciton.writeInverseDielectricMatrix("../hBN_DFT_HSE06_q_inv_epsilon.dat");

    hBN_exciton.writeDielectricMatrix("../hBN_DFT_HSE06_q_epsilon.dat");

    // Testing the code for RPA calculation
    // hBN_exciton.compute_2D_RPAInvDielectricMatrix("MoS2_TB_q_points_test.dat");

    // hBN_exciton.writeRPAPolarizabilityMatrix("MoS2_RPA_polarizability.dat");

    // hBN_exciton.writeRPAInverseDielectricMatrix("MoS2_TB_RPA_screening.dat");

    return 0;
}