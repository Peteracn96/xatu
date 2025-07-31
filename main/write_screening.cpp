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

int main(int argc, char* argv[]){

    // Parse console stdin

    if (argc != 5){
		throw std::invalid_argument("Error: 4 input files are expected");
	}
    else if (argc < 5){
        throw std::invalid_argument("Error: At least two input file are required (system config, exciton config, screening config and q points file).");
    };

    std::string modelfile = argv[1];
    std::string excitonfile = argv[2];
    std::string screeningfile = argv[3];
    std::string q_points_file = argv[4];
    
    std::unique_ptr<xatu::SystemConfiguration> systemConfig;
    std::unique_ptr<xatu::ExcitonConfiguration> excitonConfig;
    std::unique_ptr<xatu::ScreeningConfiguration> screeningConfig;

    if (modelfile.find(".outp") != std::string::npos){
        systemConfig.reset(new xatu::CRYSTALConfiguration(modelfile, 100));
    }
    else if (modelfile.find(".model") != std::string::npos){
        systemConfig.reset(new xatu::SystemConfiguration(modelfile));
    } else {
        throw std::invalid_argument("Error: Unsupported system configuration file format. Use .outp or .model files.");
    }


    screeningConfig.reset(new xatu::ScreeningConfiguration(screeningfile));
    excitonConfig.reset(new xatu::ExcitonConfiguration(excitonfile));


    xatu::ExcitonTB exciton = xatu::ExcitonTB(*systemConfig, *excitonConfig, *screeningConfig);
    exciton.setMode(excitonConfig->excitonInfo.mode);

    if (modelfile.find(".outp") != std::string::npos){
        exciton.system->setAU(true); // if input model is CRYSTAL
    }
    
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    std::cout << "|                                  Parameters                               |" << std::endl;
    std::cout << "+---------------------------------------------------------------------------+" << std::endl;
    
    std::cout << "System configuration file: " << modelfile << "\n" << std::endl;
    std::cout << "q points file file:        " << q_points_file << "\n" << std::endl;


    exciton.brillouinZoneMesh(exciton.ncell);
    exciton.initializeHamiltonian();

    exciton.writeBZtofile();

    // const int array_size = 11;
    // int nGs_array[array_size] = {1,8,9,13,19,30,37,39,43,55,61};

    // for (int gs = 0; gs < array_size; ++gs) {

	//     int nGs = nGs_array[gs];

	//     exciton.setReciprocalVectors(nGs);

    //     // Testing the code for RPA calculation
    //     exciton.compute_2D_RPAInvDielectricMatrix("MoS2_TB_q_points_test.dat");

    //     exciton.writeRPAPolarizabilityMatrix("MoS2_RPA_polarizability_" + std::to_string(nGs) + ".dat");

    //     exciton.writeRPAInverseDielectricMatrix("MoS2_TB_RPA_screening_" + std::to_string(nGs) + ".dat");
    // }

    //exciton.compute_2D_DielectricMatrix_Opt("MoS2_TB_q_points_test.dat"); not optimal at the end

    exciton.compute_2D_DielectricMatrix(q_points_file);

    size_t lastindex = q_points_file.find_last_of("."); 
    std::string rawname = q_points_file.substr(0, lastindex); 

    exciton.invertDielectricMatrix();

    // exciton.writePolarizabilityMatrix(rawname + "_polarizability.dat");

    // exciton.writeDielectricMatrix(rawname + "_epsilon.dat");

    exciton.writeInverseDielectricMatrix(rawname + "_inv_epsilon.dat");

    

    return 0;
}