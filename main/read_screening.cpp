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

    int NGs = mos2_exciton.getNGs();

    std::cout << "Area of unit cell = " << mos2_exciton.system->unitCellArea << std::endl;

    mos2_exciton.readInverseDielectricMatrix("MoS2_TB_screening_43Gs.dat");

    return 0;
}