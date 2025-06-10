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

// example of run command: ./rpa_screening ../examples/material_models/MoS2.model ../examples/excitonconfig/MoS2_test.txt ../examples/screeningconfig/MoS2_TB_screening.txt

int main(int argc, char *argv[])
{

    //auto start = high_resolution_clock::now();

    // Parse console stdin

    if (argc != 4)
    {
        throw std::invalid_argument("Error: Two input files are expected");
    }
    else if (argc < 4)
    {
        throw std::invalid_argument("Error: At least two input file are required (system config, exciton config and screening config).");
    };

    std::string modelfile = argv[1];
    std::string screeningfile = argv[3];
    std::string excitonfile = argv[2];

    std::cout << "System configuration file parsed." << std::endl;

    // xatu::ExcitonTB bulkExciton = xatu::ExcitonTB(*systemConfig, ncell, nbands, nrmbands, parameters, Q, Gcutoff, nG);

    std::cout << "Exciton configuration initialized." << std::endl;

    xatu::SystemConfiguration model_config(modelfile);

    xatu::ExcitonConfiguration exciton_config(excitonfile);

    xatu::ScreeningConfiguration screening_config(screeningfile);

    xatu::ExcitonTB mos2_exciton(model_config, exciton_config,screening_config);

    mos2_exciton.brillouinZoneMesh(mos2_exciton.ncell);
    mos2_exciton.initializeHamiltonian();

    int NGs = mos2_exciton.nReciprocalVectors;
    uint nGs_aux = 0;

    std::cout << "Number of reciprocal vectors used in the calculation: " << NGs << std::endl;

    
    arma::vec Gcutoff_array = arma::regspace(3., 3., 45.0);
    const int array_size = Gcutoff_array.n_rows;

   //  int nGs_array[array_size] = {7,9,13,19,30,37,39,43,55,61};

    for (int gs = 0; gs < array_size; ++gs) {

        double gc = Gcutoff_array(gs);
        mos2_exciton.setGcutoff(gc);

        // int nGs = nGs_array[gs];

        // If the new cutoff generates the same number of Gs as the previous iteration, increment i
        if (nGs_aux == mos2_exciton.getNGs())
        {
            ++gs;
            continue;
        }

        nGs_aux = mos2_exciton.getNGs();
        std::cout << "nGs: " << mos2_exciton.getNGs() << std::endl;

        mos2_exciton.setReciprocalVectors(nGs_aux);

        // Testing the code for RPA calculation
        mos2_exciton.compute_2D_RPAInvDielectricMatrix("MoS2_TB_q_points_test.dat");

        mos2_exciton.writeRPAPolarizabilityMatrix("MoS2_TB_RPA_polarizability_" + std::to_string(nGs_aux) + "test.dat");

        mos2_exciton.writeRPAInverseDielectricMatrix("MoS2_TB_RPA_screening_" + std::to_string(nGs_aux) + "test.dat");
    }

    return 0;
}