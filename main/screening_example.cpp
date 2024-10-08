#include <armadillo>
#include <xatu.hpp>


int main(){

    arma::rowvec q = {1.2, 0.3, 0};

    int vbands = 34;
    int cbands = 16;
    int Ncells = 15;
    double Gcutoff = 25;

    auto model_config = xatu::SystemConfiguration("../examples/material_models/wannier_models/MoS2_spin_wannier_07032024.model");

    auto exciton_config = xatu::ExcitonConfiguration("../examples/excitonconfig/MoS2_test.txt");

    auto screening_config = xatu::ScreeningConfiguration("../examples/screeningconfig/MoS2_wannier_screening.txt");

    auto mos2_exciton = xatu::ExcitonTB(model_config, exciton_config, screening_config);

    mos2_exciton.brillouinZoneMesh(mos2_exciton.ncell);
    mos2_exciton.initializeHamiltonian();

    mos2_exciton.computesingleDielectricFunction();
    
    return 0;
}