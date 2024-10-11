#include <armadillo>
#include <xatu.hpp>


int main(){

    arma::rowvec q = {1.2, 0.3, 0};

    int nstates = 8;
    int decimals = 6;

    auto model_config = xatu::SystemConfiguration("../examples/material_models/MoS2.model");

    auto exciton_config = xatu::ExcitonConfiguration("../examples/excitonconfig/MoS2_test.txt");

    //auto screening_config = xatu::ScreeningConfiguration("../examples/screeningconfig/MoS2_TB_screening.txt");

    auto mos2_exciton = xatu::ExcitonTB(model_config, exciton_config);

    mos2_exciton.brillouinZoneMesh(mos2_exciton.ncell);
    mos2_exciton.initializeHamiltonian();


    mos2_exciton.setVectors(0,0);
    //mos2_exciton.computesingleDielectricFunction();

    std::cout << std::left << std::setw(30) << "Dielectric constant of embedding medium: " << mos2_exciton.eps_m << std::endl;
    std::cout << std::left << std::setw(30) << "Dielectric constant of substrate: " << mos2_exciton.eps_s << std::endl;

    FILE* energies_file = fopen("mos2_exciton_nGs.txt", "w");

    int NGs = mos2_exciton.getNGs();

    int nGsarray[] = {7,13,19,31,37,43};

    for (int ngs : nGsarray){
        mos2_exciton.setReciprocalVectors(ngs);
        mos2_exciton.BShamiltonian();

        auto results = mos2_exciton.diagonalize("diag", nstates);

        fprintf(energies_file, "%d\t ", ngs);

        for(int i = 0; i < nstates; i++){
            fprintf(energies_file, "%f\t ", results->eigval(i));
        }
        fprintf(energies_file, "\n");
    }
    
    return 0;
}