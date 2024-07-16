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
        // Number of conduction bands removed from calculation
        int nrmcbands = 0;
        // Momentum to compute the dielectric function.
        arma::rowvec q = {0.2, 0., 0.};
        // Cut off for G vectors
        double Gcutoff;
        // Bool true/false if Gcutoff was/was not found
        bool Gcutoff_found = false;
        // Reduction factor of the BZ mesh. Defaults to 1.
        int submeshFactor = 1;
        // Calculation mode (either 'realspace' or 'reciprocalspace')
        std::string mode = "reciprocalspace";
        // Scissor cut to correct the bandgap
        double scissor = 0.0;
        // Number of reciprocal vectors to use in reciprocal space calculation
        int nReciprocalVectors = 0;
        // Pair of reciprocal vectors (G,G') to compute the dielectric function at
        arma::ivec Gs = {0, 0};
        // Number of Bravais lattice vectors to use in real space calculation
        int nLatticeVectors = 0;
        // Regularization distance
        double regularization = 0.0;
        // Compute dielectric function or polarizability or none (in case of none it computes the exciton)
        std::string function = "none";
        // Bool that is true/false when the dielectric function is/is not used
        bool isscreeningset = "false";
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