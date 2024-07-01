#include <fstream>
#include <iostream>
#include "xatu/ScreeningConfiguration.hpp"

namespace xatu {

/**
 * Default constructor.
 * @details Throws an error if called. Class must be constructed providing a filename with the configuration. 
 */
ScreeningConfiguration::ScreeningConfiguration(){
    throw std::invalid_argument("Error: ScreeningConfiguration must be invoked with one argument (filename)");
};

/**
 * File constructor. 
 * @details ScreeningConfiguration must be initialized always with this constructor.
 * Upon call, the exciton configuration file is fully parsed and the information extracted.
 * @param filename Name of file with the exciton configuration.
 */
ScreeningConfiguration::ScreeningConfiguration(std::string filename) : ConfigurationBase(filename){
    this->expectedArguments = {"mode","removedbands"};
    parseContent();
    checkArguments();
    checkContentCoherence();
}

/**
 * Method to parse the exciton configuration from its file.
 * @details This method extracts all information from the configuration file and
 * stores it with the adequate format in the information struct. 
 */
void ScreeningConfiguration::parseContent(){
    extractArguments();
    extractRawContent();

    if (contents.empty()){
        throw std::logic_error("File contents must be extracted first");
    }
    for(const auto& arg : foundArguments){
        auto content = contents[arg];

        if (content.size() == 0){
            continue;
        }
        else if(content.size() != 1){
            throw std::logic_error("Expected only one line per field");
        }
        else if(arg == "valencebands"){
            screeningInfo.nvbands = parseScalar<int>(content[0]);
        }
        else if(arg == "conductionbands"){
            screeningInfo.ncbands = parseScalar<int>(content[0]);
        }
        else if(arg == "removedbands"){
            screeningInfo.nrmcbands = parseScalar<int>(content[0]);
        }
        else if(arg == "reciprocalvectors"){
            std::vector<int> gs = parseLine<int>(content[0]);
            screeningInfo.Gs(0) = gs[0];
            screeningInfo.Gs(1) = gs[1];
        }
        else if(arg == "momentum"){
            std::vector<double> momentum = parseLine<double>(content[0]);
            screeningInfo.q = arma::rowvec(momentum);
        }
        else if(arg == "Gcutoff"){
            screeningInfo.Gcutoff = parseScalar<double>(content[0]);
        }
        else if(arg == "mode"){
            screeningInfo.mode = content[0];
            if (screeningInfo.mode == "reciprocalspace"){
                screeningInfo.nReciprocalVectors = parseScalar<int>(content[0]);
            } else if (screeningInfo.mode == "realspace") {
                screeningInfo.nLatticeVectors = parseScalar<int>(content[0]);
            } else {
                throw std::invalid_argument("Mode not recognized, must be 'realspace' or 'reciprocalspace'. Exiting.");
            }
                
        }
        else if(arg == "regularization"){
            screeningInfo.regularization = parseScalar<double>(content[0]);
        }
        else{    
            std::cout << "Unexpected argument: " << arg << ", skipping block..." << std::endl;
        }
    }
};

/**
 * Method to check whether the information extracted from the configuration file is
 * consistent and well-defined. 
 */
void ScreeningConfiguration::checkContentCoherence(){
    if(screeningInfo.ncell <= 0 && screeningInfo.mode == "realspace"){
        throw std::logic_error("'ncell' must be a positive number");
    };
    if(screeningInfo.nvbands <= 0){
        throw std::logic_error("'nvbands' must be a positive number");
    };
    if(screeningInfo.ncbands <= 0){
        throw std::logic_error("'ncbands' must be a positive number");
    }
    if(screeningInfo.nrmcbands < 0){
        throw std::logic_error("'nrmcbands' must not be a negative number");
    }
    if(screeningInfo.nReciprocalVectors < 0){
        throw std::logic_error("'nReciprocalVectors' must be a non-negative number");
    }
    if(screeningInfo.nLatticeVectors <= 0 && screeningInfo.mode == "realspace"){
        throw std::logic_error("'nLatticeVectors' must be a positive number");
    }
    if (screeningInfo.mode != "realspace" && screeningInfo.mode != "reciprocalspace"){
        throw std::invalid_argument("Invalid mode. Use 'realspace' or 'reciprocalspace'");
    }
};

}