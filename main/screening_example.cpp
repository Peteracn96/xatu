#include <armadillo>
#include <xatu.hpp>


int main(){

    arma::rowvec q = {1.2, 0.3, 0};

    int vbands = 34;
    int cbands = 16;
    int Ncells = 15;
    double Gcutoff = 25;

    auto model_config = xatu::SystemConfiguration("../examples/material_models/wannier_models/MoS2_spin_wannier_07032024.model");

    auto exciton_config = xatu::ExcitonConfiguration("../examples/excitonconfig/MoS2_wannier.txt");

    auto screening_config = xatu::ScreeningConfiguration("../examples/screeningconfig/MoS2_wannier_screening.txt");

    auto mos2_exciton = xatu::ExcitonTB(model_config, exciton_config, screening_config);

    mos2_exciton.brillouinZoneMesh(mos2_exciton.ncell);
    mos2_exciton.initializeHamiltonian();


    mos2_exciton.setVectors(0,0);
    //mos2_exciton.computesingleDielectricFunction();

    // Prints the k points to plot the matrix elements in a grid
    std::string filename_k_points = "kgrid_Wannier" + std::to_string(mos2_exciton.ncell) + ".dat";

    std::ofstream k_points_file; 

    k_points_file.open(filename_k_points);

    if (!k_points_file.is_open()) { // check if the file was opened successfully
        std::cerr << "Error opening file\n";
        std::cerr << errno << "\n";
    }


    for(unsigned int i = 0; i < mos2_exciton.ncell*mos2_exciton.ncell; i++){
        auto k = mos2_exciton.system->kpoints.row(i);
        k_points_file << k(0) << " " << k(1) << " " << k(2) << std::endl;
    }

    k_points_file.close();
    
    return 0;
}