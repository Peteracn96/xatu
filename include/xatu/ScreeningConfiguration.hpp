#pragma once
#include <armadillo>
#include "xatu/ConfigurationBase.hpp"

namespace xatu {

/**
 * ScreeningConfiguration is a specialization of ConfigurationBase to parse
 * dielectric function configuration files. 
 */
class ScreeningConfiguration : public ConfigurationBase{

    struct configuration {
        // Simulation label
        std::string label;
        // Number of unit cells in the calculation
        int ncell;
        // Number of valence bands included calculation.
        int nvbands = 0;
        // Number of conduction bands included calculation.
        int ncbands = 0;
        // Cut off for G vectors
        double Gcutoff;
        // Reduction factor of the BZ mesh. Defaults to 1.
        int submeshFactor = 1;
        // Calculation mode (either 'realspace' or 'reciprocalspace')
        std::string mode = "reciprocalspace";
        // Scissor cut to correct the bandgap
        double scissor = 0.0;
        // Number of reciprocal vectors to use in reciprocal space calculation
        int nReciprocalVectors = 0;
        // Number of Bravais lattice vectors to use in real space calculation
        int nLatticeVectors = 0;
        // Regularization distance
        double regularization = 0.0;
    };

    public:
        configuration screeningInfo;
    
    public:
        ScreeningConfiguration();
        ScreeningConfiguration(std::string);
    
    private:
        virtual void parseContent();
        void checkContentCoherence();
};

}