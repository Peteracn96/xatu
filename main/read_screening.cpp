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

    int nstates = 8;
    int decimals = 6;

    xatu::SystemConfiguration model_config("../examples/material_models/MoS2.model");

    xatu::ExcitonConfiguration exciton_config("../examples/excitonconfig/MoS2_test.txt");

    xatu::ScreeningConfiguration screening_config("../examples/screeningconfig/MoS2_TB_screening.txt");

    xatu::ExcitonTB mos2_exciton(model_config, exciton_config,screening_config);

    mos2_exciton.brillouinZoneMesh(mos2_exciton.ncell);
    mos2_exciton.initializeHamiltonian();

    int NGs = mos2_exciton.nReciprocalVectors;

    std::cout << "Number of reciprocal vectors used in the calculation: " << NGs << std::endl;

    mos2_exciton.readInverseDielectricMatrix("MoS2_TB_screening_43Gs.dat");

    mos2_exciton.BShamiltonian();

    auto results = mos2_exciton.diagonalize("diag", nstates);

    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                                    Results                                |" << std::endl;
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;

    xatu::printEnergies(results, nstates, decimals);

    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                                    Output                                 |" << std::endl;
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;

    std::string output = exciton_config.excitonInfo.label;

    // --------------------------- Output ---------------------------
    bool writeEigvals = true;
    if(writeEigvals){
        std::string filename_en = output + ".eigval";
        FILE* textfile_en = fopen(filename_en.c_str(), "w");

        std::cout << "Writing eigvals to file: " << filename_en << std::endl;
        fprintf(textfile_en, "%d\n", exciton_config.excitonInfo.ncell);
        results->writeEigenvalues(textfile_en, nstates);

        fclose(textfile_en);
    }
    
    bool writeStates = true;
    if(writeStates){
        std::string filename_st = output + ".states";
        FILE* textfile_st = fopen(filename_st.c_str(), "w");

        std::cout << "Writing states to file: " << filename_st << std::endl;
        results->writeStates(textfile_st, nstates);

        fclose(textfile_st);
    }
    
    bool writeWF = true;
    if(writeWF){
        std::string filename_kwf = output + ".kwf";
        FILE* textfile_kwf = fopen(filename_kwf.c_str(), "w");

        std::cout << "Writing k w.f. to file: " << filename_kwf << std::endl;
        for(int stateindex = 0; stateindex < nstates; stateindex++){
            if (exciton_config.excitonInfo.submeshFactor != 1){
                results->writeReciprocalAmplitude(stateindex, textfile_kwf);
            }
            else{
                results->writeExtendedReciprocalAmplitude(stateindex, textfile_kwf);
            }
        }

        fclose(textfile_kwf);
    }
    
    bool writeAbs = true;
    if(writeAbs){
        std::cout << "Writing absorption spectrum fo file... " << std::endl;
        results->writeAbsorptionSpectrum();
    }

    return 0;
}