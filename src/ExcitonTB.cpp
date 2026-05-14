#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include "xatu/ExcitonTB.hpp"
#include "xatu/utils.hpp"
#include "xatu/davidson.hpp"
#include <set>
using namespace arma;
using namespace std::chrono;

/** 
 *Method to prompt user to continue
 *@param message
 *@return void 
*/
void continueprompt(std::string message){

    std::string to_continue = "_";

    while(to_continue != "y" && to_continue != "n"){
        std::cout << message;
        std::getline(std::cin, to_continue);
        if (to_continue == "n"){
            std::cout << "You have chosen not to continue. Exiting.\n";
            exit(1);
        } else if (to_continue == "y") {
            continue;
        } else {
            std::cout << "Option not recognized. Please enter 'y' or 'n' (without the ticks).\n";
            continue;
        }
    }
}

namespace xatu {

/**
 * Method to set the values of the attributes of an exciton object.
 * @param ncell Number of unit cells per axis.
 * @param bands Vector with the indices of the bands that form the exciton.
 * @param parameters Dielectric constants and screening length.
 * @param Q Center-of-mass momentum of the exciton.
 * @return void 
 */
void ExcitonTB::initializeExcitonAttributes(int ncell, const arma::ivec& bands, 
                                      const arma::rowvec& parameters, const arma::rowvec& Q){
    this->ncell_ = ncell;
    this->totalCells_ = pow(ncell, system_->ndim);
    this->Q_          = Q;
    this->bands_      = bands;
    this->eps_m_      = parameters(0);
    this->eps_s_      = parameters(1);
    this->r0_         = parameters(2);
    this->cutoff_     = ncell/2.5; // Default value, seems to preserve crystal point group
    if(r0 == 0){
        throw std::invalid_argument("Error: r0 must be non-zero");
    }
}

/**
 * Method to set the attributes of an exciton object from a ExcitonConfiguration object.
 * @details Overload of the method to use a configuration object. Based on the parametric method.
 * @param cfg ExcitonConfiguration object from parsed file.
 * @return void
 */
void ExcitonTB::initializeExcitonAttributes(const ExcitonConfiguration& cfg){

    int ncell        = cfg.excitonInfo.ncell;
    int nbands       = cfg.excitonInfo.nbands;
    arma::ivec bands = cfg.excitonInfo.bands;
    arma::rowvec parameters = {cfg.excitonInfo.eps(0), cfg.excitonInfo.eps(1), cfg.excitonInfo.eps(2)};
    arma::rowvec Q   = cfg.excitonInfo.Q;
    double percentage = cfg.excitonInfo.percentage;

    if (2*nbands > system->basisdim){
        std::cout << "Error: Number of bands cannot be higher than actual material bands" << std::endl;
        exit(1);
    }

    if (bands.empty()){
        bands = arma::regspace<arma::ivec>(- nbands + 1, nbands);
    }

    initializeExcitonAttributes(ncell, bands, parameters, Q);

    std::vector<arma::s64> valence, conduction;
    for(uint i = 0; i < bands.n_elem; i++){
        if (bands(i) <= 0){
            valence.push_back(bands(i) + system->fermiLevel);
        }
        else{
            conduction.push_back(bands(i) + system->fermiLevel);
        }
    }
    this->valenceBands_ = arma::ivec(valence);
    this->conductionBands_ = arma::ivec(conduction);
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));

    // Set flags
    this->exchange = cfg.excitonInfo.exchange;
    this->scissor_ = cfg.excitonInfo.scissor;
    this->mode_    = cfg.excitonInfo.mode;
    this->Gc_exciton_ = cfg.excitonInfo.Gc_ReciprocalVectors;
    this->regularization_ = cfg.excitonInfo.regularization;
    if (regularization_ == 0){
        regularization_ = system->a;
    }
    this->potential_ = cfg.excitonInfo.potential;
    this->exchangePotential_ = cfg.excitonInfo.exchangePotential;

    this->percentage_ = percentage;

    if (cfg.excitonInfo.Gcutoff_found){
        this->Gc_exciton_ = cfg.excitonInfo.Gc_ReciprocalVectors;
    } else {
        this->Gc_exciton_ = (this->ncell)/2.5 * arma::norm(system->reciprocalLattice.row(0));
    }

    if (this->mode == "reciprocalspace"){
        if (this->trunreciprocalLattice_.is_empty()){

            this->trunreciprocalLattice_ = system_->truncateReciprocalSupercell(this->Gc_exciton_);

            this->nReciprocalVectors_ = this->trunreciprocalLattice_.n_rows;

            if (this->nReciprocalVectors_ > (int)this->trunreciprocalLattice_.n_rows){
                throw std::invalid_argument("initializeExcitonAttributes(): Number of reciprocal lattice vectors for the exciton (" + std::to_string(this->nReciprocalVectors_) + ") may not exceed the number of vectors included in the screening (" + std::to_string(this->trunreciprocalLattice_.n_rows) + ") .");
            }
        }
    } else if (this->mode == "realspace"){
        std::cout << "Doing nothing for now\n";
    }
    
}

/*
 * Method to verify if the chosen potential was 'rpa' while a screening file was provided or not
 * @return void
*/
void ExcitonTB::verifypotential(){
    if ((this->potential_ == "rpa" || this->exchangePotential_ == "rpa") && this->isscreeningset == false) {
        std::cout << "To use the rpa potential, a screening configuration file must be provided." << std::endl;
        exit(1);
    }

    if ((this->potential_ != "rpa" || (this->exchangePotential_ != "rpa" && this->exchange)) && this->isscreeningset == true && this->function_ == "exciton") {
        // continueprompt("You have provided a screening file, yet you have not chosen the rpa potential.\nDo you wish to continue?[y/n]\n");
        std::cout << "You have provided a screening file, yet you have not chosen the rpa potential." << std::endl;
        exit(1);
    }
}

/**
 * Method to set the screening attributes of an exciton object from a ScreeningConfiguration object.
 * @details Overload of the method to use a configuration object. Based on the parametric method.
 * @param cfg ScreeningConfiguration object from parsed file.
 * @param mode Real space or reciprocal space mode
 * @return void
 */
void ExcitonTB::initializeScreeningAttributes(const ScreeningConfiguration& cfg){
    uint nvalencebands        = cfg.screeningInfo.nvbands;
    uint nconductionbands     = cfg.screeningInfo.ncbands;
    arma::rowvec q            = cfg.screeningInfo.q;
    arma::ivec gs             = cfg.screeningInfo.Gs;
    std::string function      = cfg.screeningInfo.function;
    arma::ivec ts             = cfg.screeningInfo.ts;
    uint ncell_aux            = cfg.screeningInfo.ncell_aux;
    bool model_has_spin       = cfg.screeningInfo.spin;
    bool isotropic            = cfg.screeningInfo.isotropic;
    double Gcutoff            = cfg.screeningInfo.Gcutoff;
    double d                  = cfg.screeningInfo.d;
    this->ts_ = ts;

    this->isscreeningset      = true;
    this->function_           = function;
    this->ncell_aux_          = ncell_aux;
    this->nk_aux_             = pow(ncell_aux, 2);
    this->Gcutoff_            = Gcutoff;
    this->d_                  = d;

    uint totalvbands = system->fermiLevel + 1;
    uint totalcbands = system->basisdim - totalvbands;

    if (nvalencebands > totalvbands  || nconductionbands > totalcbands){
        std::cout << "Number of included bands cannot be higher than the number of material bands" << std::endl;
        std::cout << "Number of total valence bands = " << totalvbands<< std::endl;
        std::cout << "Number of total conduction bands = " << totalcbands << std::endl;
    
        exit(1);
    }

    if (nvalencebands + nconductionbands > (uint)system->basisdim){
        std::cout << "Error: Number of bands cannot be higher than actual material" << std::endl;
        std::cout << "Total number of bands is " << system->basisdim << std::endl;
        exit(1);
    }

    this->q_ = q;

    this->nvalencebands_ = nvalencebands;
    this->nconductionbands_ = nconductionbands;

    this->Gs_ = gs;

    if (model_has_spin) {
        this->g_s = 1;
    } else {
        this->g_s = 2;
    }

    std::vector<arma::s64> valence, conduction;
    int basisdim = system->basisdim;

    for(int i = 0; i < basisdim; i++){
        if (i <= system->fermiLevel){
            valence.push_back(i);
        }
        else{
            conduction.push_back(i);
        }
    }
    
    this->valencebands_ = arma::ivec(valence);
    this->conductionbands_ = arma::ivec(conduction);

    std::cout << "Creating the coarser BZ mesh... " << std::flush;

    int ndim = 2;
    arma::mat kpoints(pow(ncell_aux, ndim), 3);
    arma::mat combinations = system->generateCombinations(ncell_aux, ndim);
    if (ncell_aux % 2 == 1)
    {
        combinations += 1. / 2;
    }

    for (uint i = 0; i < nk_aux; i++)
    {
        arma::rowvec kpoint = arma::zeros<arma::rowvec>(3);
        for (int j = 0; j < ndim; j++)
        {
            kpoint += (2 * combinations.row(i)(j) - ncell_aux) / (2 * ncell_aux) * system->reciprocalLattice_.row(j);
        }
        kpoints.row(i) = kpoint;
    }

    this->kpoints_aux_ = kpoints;
    this->nk_aux_ = kpoints.n_rows;

    std::cout << "Done. Number of k points in the coarser BZ mesh: " << nk_aux << "\n" << std::endl;

    this->eigveckStack_  = arma::cx_cube(basisdim, basisdim, nk_aux); // For the dielectric function
    this->eigvalkStack_  = arma::mat(basisdim, nk_aux);

    if (this->mode == "realspace"){
        if (cfg.screeningInfo.function == "exciton"){
            std::cout << "Computation of the exciton using the RPA dielectric function in real space not available yet." << std::endl;
        }
    } else if (this->mode == "reciprocalspace"){
        double radius = this->Gcutoff_;
        this->trunreciprocalLattice_ = system_->truncateReciprocalSupercell(radius);

        uint ngs = this->nGs = this->trunreciprocalLattice_.n_rows;

        if (ngs < this->nReciprocalVectors) {
            throw std::invalid_argument("initializeExcitonAttributes(): Number of reciprocal lattice vectors for the screening (" + std::to_string(ngs) + ") may not be less than the number of vectors included for the exciton (" + std::to_string(this->nReciprocalVectors_) + ") .");
        }

        if (cfg.screeningInfo.function == "dielectric" || cfg.screeningInfo.function == "polarizability"){
            if (gs(0) >= ngs || gs(1) >= ngs){
                std::cout << "Error: Index of the reciprocal vector must not be higher than number of reciprocal vectors" << std::endl;
                std::cout << "Number of reciprocal vectors is " << ngs << std::endl;

                exit(1);
            }
        }

        if (cfg.screeningInfo.function == "exciton"){
            int nks = this->ncell*this->ncell;
            this->Chimatrix_ = arma::cx_cube(ngs, ngs, nks, arma::fill::zeros);
            this->epsilonmatrix_ = arma::cx_cube(ngs, ngs, nks, arma::fill::zeros);
            this->Invepsilonmatrix_ = arma::cx_cube(ngs, ngs, nks, arma::fill::zeros);
        }
    }
}

/**
 * Method to set the screening attributes of an exciton object from a ScreeningConfiguration object and the mode specified.
 * @details Overload of the method to use a configuration object. Based on the parametric method.
 * @param cfg ScreeningConfiguration object from parsed file.
 * @param mode Real space or reciprocal space mode
 * @return void
 */
void ExcitonTB::initializeScreeningAttributes(const ScreeningConfiguration& cfg, const std::string mode){
    
    this->initializeScreeningAttributes(cfg);
    this->setMode(mode);
}

/**
 * Exciton constructor from a SystemConfiguration object and a vector with the bands that form
 * the exciton, as well as the other parameters.
 * @param config SystemConfiguration object from config file.
 * @param ncell Number of unit cells along each axis.
 * @param bands Vector with the indices of the bands that form the exciton.
 * @param parameters Vector with dielectric constants and screening length.
 * @param Q Center-of-mass momentum.
 */
ExcitonTB::ExcitonTB(const SystemConfiguration& config, int ncell, const arma::ivec& bands, 
                     const arma::rowvec& parameters, const arma::rowvec& Q) {

    system_.reset(new SystemTB(config));

    initializeExcitonAttributes(ncell, bands, parameters, Q);

    if (bands.n_elem > excitonbasisdim){
        std::cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
        exit(1);
    }

    // arma::ivec is implemented with typedef s64
    std::vector<arma::s64> valence, conduction;
    for(uint i = 0; i < bands.n_elem; i++){
        if (bands(i) <= 0){
            valence.push_back(bands(i) + system->fermiLevel);
        }
        else{
            conduction.push_back(bands(i) + system->fermiLevel);
        }
    }
    this->valenceBands_ = arma::ivec(valence);
    this->conductionBands_ = arma::ivec(conduction);
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));
};


/**
 * Exciton constructor from a SystemConfiguration object. One specifies the number of valence and conduction
 * bands, as well as the other parameters.
 * @param config SystemConfiguration object from config file.
 * @param ncell Number of unit cells along each axis.
 * @param nbands Number of bands (same number for both valence and conduction) that form the exciton.
 * @param nrmbands Number of bands to be removed with respect to the Fermi level. 
 * @param parameters Vector with dielectric constants and screening length.
 * @param Q Center-of-mass momentum.
 */
ExcitonTB::ExcitonTB(const SystemConfiguration& config, int ncell, int nbands, int nrmbands, 
                     const arma::rowvec& parameters, const arma::rowvec& Q) : 
                     ExcitonTB(config, ncell, {}, parameters, Q){
    
    if (2*nbands > system->basisdim){
        std::cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
        exit(1);
    }

    int fermiLevel = system->fermiLevel;
    this->valenceBands_ = arma::regspace<arma::ivec>(fermiLevel - nbands - nrmbands + 1, 
                                                     fermiLevel - nrmbands);
    this->conductionBands_ = arma::regspace<arma::ivec>(fermiLevel + 1 + nrmbands, 
                                                        fermiLevel + nbands + nrmbands);
    this->bands_ = arma::join_cols(valenceBands, conductionBands) - fermiLevel;
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));
};

/**
 * Exciton constructor from a SystemConfiguration object and a vector with the bands that form
 * the exciton, as well as the other parameters.
 * @param config SystemConfiguration object from config file.
 * @param ncell Number of unit cells along each axis.
 * @param bands Vector with the indices of the bands that form the exciton.
 * @param parameters Vector with dielectric constants and screening length.
 * @param Q Center-of-mass momentum.
 */
ExcitonTB::ExcitonTB(const SystemConfiguration& config, int ncell, int nbands, int nrmbands,
                     const arma::rowvec& parameters, const arma::rowvec& Q, const double Gcutoff, const double Gc_exciton) : ExcitonTB(config, ncell, {}, parameters, Q)
{
    if (2 * nbands > system->basisdim)
    {
        std::cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
        exit(1);
    }

    int fermiLevel = system->fermiLevel;
    this->valenceBands_ = arma::regspace<arma::ivec>(fermiLevel - nbands - nrmbands + 1,
                                                     fermiLevel - nrmbands);
    this->conductionBands_ = arma::regspace<arma::ivec>(fermiLevel + 1 + nrmbands,
                                                        fermiLevel + nbands + nrmbands);
    this->bands_ = arma::join_cols(valenceBands, conductionBands) - fermiLevel;
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));
    this->Gcutoff_ = Gcutoff;
    this->Gc_exciton_ = Gc_exciton;

    this->nReciprocalVectors_ = system_->truncateReciprocalSupercell(this->Gc_exciton_).n_rows;
    this->trunreciprocalLattice_ = system_->truncateReciprocalSupercell(this->Gcutoff_);

    if (this->nReciprocalVectors_ > this->trunreciprocalLattice_.n_rows) {
        throw std::invalid_argument("constructor ExcitonTB(): Number of reciprocal lattice vectors for the exciton (" + std::to_string(this->nReciprocalVectors_) + ") may not exceed the number of vectors included in the screening (" + std::to_string(this->trunreciprocalLattice_.n_rows) + ") .");
    }
};

/**
 * Exciton constructor from SystemConfiguration and ExcitonConfiguration.
 * @param config SystemConfiguration object.
 * @param excitonConfig ExcitonConfiguration object.
 */ 
ExcitonTB::ExcitonTB(const SystemConfiguration& config, const ExcitonConfiguration& excitonConfig){

    system_.reset(new SystemTB(config));
    initializeExcitonAttributes(excitonConfig);
}

/**
 * Exciton constructor from SystemConfiguration, ScreeningConfiguration and ExcitonConfiguration.
 * @param config SystemConfiguration object.
 * @param excitonConfig ExcitonConfiguration object.
 * @param screeningConfig ExcitonConfiguration object.
 */ 
ExcitonTB::ExcitonTB(const SystemConfiguration& config, const ExcitonConfiguration& excitonConfig, const ScreeningConfiguration& screeningConfig){

    system_.reset(new SystemTB(config));
    initializeExcitonAttributes(excitonConfig);
    initializeScreeningAttributes(screeningConfig, excitonConfig.excitonInfo.mode);
}

ExcitonTB::ExcitonTB(std::shared_ptr<SystemTB> sys, int ncell, const arma::ivec& bands, 
                     const arma::rowvec& parameters, const arma::rowvec& Q){

    
    system_ = sys;
    initializeExcitonAttributes(ncell, bands, parameters, Q);

    if ((int)bands.n_elem > system->basisdim){
        std::cout << "Error: Number of bands cannot be higher than actual material bands" << std::endl;
        exit(1);
    }

    // arma::ivec is implemented with typedef s64
    std::vector<arma::s64> valence, conduction;
    for(uint i = 0; i < bands.n_elem; i++){
        if (bands(i) <= 0){
            valence.push_back(bands(i) + system->fermiLevel);
        }
        else{
            conduction.push_back(bands(i) + system->fermiLevel);
        }
    }
    this->valenceBands_ = arma::ivec(valence);
    this->conductionBands_ = arma::ivec(conduction);
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));
};

/**
 * Exciton constructor from an already initialized System object, and all the exciton parameters.
 * @param system System object where excitons are computed.
 * @param ncell Number of unit cells along each axis.
 * @param nbands Number of bands (same for valence and conduction) that form the exciton.
 * @param nrmbands Number of bands to be removed with respect to the Fermi level. 
 * @param parameters Dielectric constant and screening length.
 * @param Q Center-of-mass momentum of the exciton.
 */
ExcitonTB::ExcitonTB(std::shared_ptr<SystemTB> sys, int ncell, int nbands, int nrmbands, 
                     const arma::rowvec& parameters, const arma::rowvec& Q) : 
                     ExcitonTB(sys, ncell, {}, parameters, Q) {
    
    if (2*nbands > system->basisdim){
        std::cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
        exit(1);
    }

    int fermiLevel = system->fermiLevel;
    this->valenceBands_ = arma::regspace<arma::ivec>(fermiLevel - nbands - nrmbands + 1, 
                                                     fermiLevel - nrmbands);
    this->conductionBands_ = arma::regspace<arma::ivec>(fermiLevel + 1 + nrmbands, 
                                                        fermiLevel + nbands + nrmbands);
    this->bands_ = arma::join_cols(valenceBands, conductionBands) - fermiLevel;
    this->bandList_ = arma::conv_to<arma::uvec>::from(arma::join_cols(valenceBands, conductionBands));
};

/** 
 * Exciton destructor.
 * @details Used mainly for debugging; the message should be removed at some point.
 */
ExcitonTB::~ExcitonTB(){};


/* ------------------------------ Setters ------------------------------ */

/**
 * Method to set the parameters of the Keldysh potential, namely the environmental
 * dielectric constants and the effective screening length.
 * @param parameters Vector with the three parameters, '{eps_m, eps_s, r0}'. 
 * @return void
 */
void ExcitonTB::setParameters(const arma::rowvec& parameters){
    if(parameters.n_elem == 3){
        eps_m_ = parameters(0);
        eps_s_ = parameters(1);
        r0_    = parameters(2);
    }
    else{
        std::cout << "parameters array must be 3d (eps_m, eps_s, r0)" << std::endl;
    }
}

/**
 * Sets the parameters of the Keldysh potential.
 * @param eps_m Dielectric constant of embedding medium.
 * @param eps_s Dielectric constant of substrate.
 * @param r0 Effective screeening length.
 * @return void 
 */
void ExcitonTB::setParameters(double eps_m, double eps_s, double r0){
    // TODO: Introduce additional comprobations regarding value of parameters (positive)
    eps_m_ = eps_m;
    eps_s_ = eps_s;
    r0_    = r0;
}

/**
 * Sets the gauge used for the Bloch basis, either 'lattice' or 'atomic'.
 * @param gauge Gause to be used, default to 'lattice'.
 * @return void
 */
void ExcitonTB::setGauge(std::string gauge){
    if(gauge != "lattice" && gauge != "atomic"){
        throw std::invalid_argument("setGauge(): gauge must be either lattice or atomic");
    }
    this->gauge_ = gauge;
}

/**
 * Sets the number of valence bands included in the computation of the screening.
 * @param nvbands
 * @return void
 */
void ExcitonTB::setValenceBands(int nvbands){
    if(nvbands <= 0){
        throw std::invalid_argument("setValenceBands(): number of valence bands must be a positive integer");
    }
    this->nvalencebands_ = nvbands;
}

/**
 * Sets the number of conduction bands included in the computation of the screening.
 * @param ncbands
 * @return void
 */
void ExcitonTB::setConductionBands(int ncbands){
    if(ncbands <= 0){
        throw std::invalid_argument("setConductionBands(): number of valence bands must be a positive integer");
    }
    this->nconductionbands_ = ncbands;
}

/**
 * Sets the type of calculation used to obtain the exciton spectrum. It can be 'realspace' (default) 
 * or 'reciprocalspace'.
 * @param mode Calculation model.
 * @return void 
 */
void ExcitonTB::setMode(std::string mode){
    if(mode != "realspace" && mode != "reciprocalspace"){
        throw std::invalid_argument("setMode(): mode must be either realspace or reciprocalspace");
    }
    this->mode_ = mode;
}

/**
 * Sets the number of reciprocal vectors to use if the exciton calculation is set to 'reciprocalspace'.
 * @param Gc_exciton Cutoff for reciprocal lattice vectors included in the calculation of the exciton
 * @return void 
 */
void ExcitonTB::setReciprocalVectors(double Gc_exciton){

    int nReciprocalVectors = system_->truncateReciprocalSupercell(Gc_exciton).n_rows;

    if(Gc_exciton < 0){

        throw std::invalid_argument("setReciprocalVectors(): given cutoff must be positive");

    } else if ((uint)nReciprocalVectors > this->trunreciprocalLattice_.n_rows && !this->trunreciprocalLattice_.is_empty()) {

        throw std::invalid_argument("setReciprocalVectors(): Number of reciprocal lattice vectors for the exciton (" + std::to_string(nReciprocalVectors) + ") may not exceed the number of vectors included in the screening (" + std::to_string(this->trunreciprocalLattice_.n_rows) + ") .");
        exit(0);

    } else if (this->trunreciprocalLattice_.is_empty()) {

        std::cout << "The reciprocal lattice vectors for calculating the exciton were not generated. Terminating." << std::endl;
        exit(0);

    }

    this->Gc_exciton_ = Gc_exciton;
    this->nReciprocalVectors_ = nReciprocalVectors;
}

/**
 * Sets the regularization distance for the Coulomb potential divergence at r=0.
 * @param regularization Distance in Angstroms.
 * @return void
*/
void ExcitonTB::setRegularization(double regularization){
    this->regularization_ = regularization;
}

/**
 * Sets the cutoff in length for truncation of the reciprocal lattice.
 * @param Gcutoff Cutoof in Angstroms^-1.
 * @return void
*/
void ExcitonTB::setGcutoff(double Gcutoff){
    if(Gcutoff < 0){
        throw std::invalid_argument("setGcutoff(): G cutoff for the screening must be positive");
    }

    uint nGs_aux = this->trunreciprocalLattice_.n_rows;
    arma::mat reciprocalLattice_old = this->trunreciprocalLattice_;

    if (Gcutoff != this->Gcutoff_) {
        this->trunreciprocalLattice_ = this->system->truncateReciprocalSupercell(Gcutoff); // Reset the truncated reciprocal lattice if Gcutoff changes
    }

    this->Gcutoff_ = Gcutoff;
    uint nGs = this->trunreciprocalLattice_.n_rows;
    this->nGs = nGs;
    this->Gc_exciton_ = Gcutoff;
    this->nReciprocalVectors_ = nGs;
    if (nGs_aux < nGs) {
        this->trunreciprocalLattice_.submat(0, 0, nGs_aux - 1, 2) = reciprocalLattice_old; // When increasing the G cutoff, conserves the order of the G vectors
    }

    printReciprocalLattice();
}

/**
 * Sets the vectors of the real/reciprocal lattice, where the polarizability will be computed at.
 * @details Receives as argument an arma::ivec object.
 * @param indeces_vec Vector containing the two indices.
 * @return void
*/
void ExcitonTB::setVectors(arma::ivec indeces_vec){
    if(indeces_vec(0) < 0 || indeces_vec(1) < 0){
        throw std::invalid_argument("setVectors(arma::ivec): Both vectors must have a positive index");
    }
    this->Gs_ = indeces_vec;
}

/**
 * Sets the vectors of the real/reciprocal lattice, where the polarizability will be computed at.
 * @param index1 Index of the first vector.
 * @param index2 Index of the second vector.
 * @return void
*/
void ExcitonTB::setVectors(int index1, int index2){
    if(index1 < 0 || index2 < 0){
        throw std::invalid_argument("setVectors(int,int): Both vectors must have a positive index");
    }
    setVectors(arma::ivec({index1, index2}));
}

/**
 * Sets the motif vectors where the polarizability will be computed at.
 * @details Receives as argument an arma::ivec object.
 * @param indeces_vec Vector containing the two indices.
 * @return void
*/
void ExcitonTB::setmotifVectors(arma::ivec indeces_vec){
    if(indeces_vec(0) < 0 || indeces_vec(1)){
        throw std::invalid_argument("setmotifVectors(arma::ivec): Both vectors must have a positive index");
    }
    this->ts_ = indeces_vec;
}

/**
 * Sets the motif vectors where the polarizability will be computed at.
 * @param index1 Index of the first vector.
 * @param index2 Index of the second vector.
 * @return void
*/
void ExcitonTB::setmotifVectors(int index1, int index2){
    if(index1 < 0 || index2 < 0){
        throw std::invalid_argument("setmotifVectors(int,int): Both vectors must have a positive index");
    }
    setmotifVectors(arma::ivec({index1, index2}));
}

/**
 * Sets the potential for the direct interaction matrix elements.
 * @param potential Type of potential to be used.
 * @return void 
 */
void ExcitonTB::setPotential(std::string potential){
    if(potential != "keldysh" && potential != "coulomb" && potential != "rpa"){
        throw std::invalid_argument("setPotential(): potential must be either keldysh, coulomb or rpa");
    }
    double test = STVH1(0.5); std::cout << "H_1(0.5) = " << test << std::endl;
    this->potential_ = potential;
}

/**
 * Sets the exchange potential for the exchange interaction matrix elements.
 * @param potential Type of potential to be used.
 * @return void 
 */
void ExcitonTB::setExchangePotential(std::string potential){
    if(potential != "keldysh" && potential != "coulomb" && potential != "rpa"){
        throw std::invalid_argument("setExchangePotential(): potential must be either keldysh, coulomb or rpa");
    }

    this->exchangePotential_ = potential;
}

/**
 * Sets the truncated lattice with given number of cells and cutoff as input.
 * @param ncell Number of cells
 * @param cutoff Cutoff
 * @return void 
 */
void ExcitonTB::setTrunLattice(int ncell, double cutoff){
    double radius = arma::norm(system->bravaisLattice.row(0)) * cutoff;
    this->trunLattice_ = system->truncateSupercell(ncell, radius);
};

/**
 * Sets the matrix that stores the q points at which we compute the dielectric matrix.
 * @param q_points Matrix storing the q points.
 * @return void
 */
void ExcitonTB::setq_points_list(arma::mat q_points)
{
    if (this->qpoints_list_.is_empty()) {
        this->qpoints_list_.set_size(q_points.n_rows, q_points.n_cols);
    }
    
    this->qpoints_list_ = q_points;
}

/**
 * Sets the percentage for regularization radius q0 for reciprocal space calculation of the exciton
 * @param percentage
 * @return void
*/
void ExcitonTB::setPercentage(double percentage){
    this->percentage_ = percentage;
}

/* ------------------------------ Getters ------------------------------ */
int ExcitonTB::getNGs() const {

    if (this->trunreciprocalLattice_.is_empty()){
        std::cout << "Matrix of reciprocal lattice vectors is empty. Will return 0.\n" << std::endl;
        return 0;
    }

    return this->trunreciprocalLattice_.n_rows;
}


/*---------------------------------------- Potentials ----------------------------------------*/

/** 
 * Purpose: Compute Struve function H0(x).
 * Source: http://jean-pierre.moreau.pagesperso-orange.fr/Cplus/mstvh0_cpp.txt 
 * @param X x --- Argument of H0(x) ( x ò 0 )
 * @param SH0 SH0 --- H0(x). The return value is written to the direction of the pointer.
*/
void ExcitonTB::STVH0(double X, double *SH0) const {
    double A0,BY0,P0,Q0,R,S,T,T2,TA0;
	int K, KM;

        S=1.0;
        R=1.0;
        if (X <= 20.0) {
           A0=2.0*X/PI;
           for (K=1; K<61; K++) {
              R=-R*X/(2.0*K+1.0)*X/(2.0*K+1.0);
              S=S+R;
              if (fabs(R) < fabs(S)*1.0e-12) goto e15;
           }
    e15:       *SH0=A0*S;
        }
        else {
           KM=int(0.5*(X+1.0));
           if (X >= 50.0) KM=25;
           for (K=1; K<=KM; K++) {
              R=-R*pow((2.0*K-1.0)/X,2);
              S=S+R;
              if (fabs(R) < fabs(S)*1.0e-12) goto e25;
           }
    e25:       T=4.0/X;
           T2=T*T;
           P0=((((-.37043e-5*T2+.173565e-4)*T2-.487613e-4)*T2+.17343e-3)*T2-0.1753062e-2)*T2+.3989422793;
           Q0=T*(((((.32312e-5*T2-0.142078e-4)*T2+0.342468e-4)*T2-0.869791e-4)*T2+0.4564324e-3)*T2-0.0124669441);
           TA0=X-0.25*PI;
           BY0=2.0/sqrt(X)*(P0*sin(TA0)+Q0*cos(TA0));
           *SH0=2.0/(PI*X)*S+BY0;
        }
}

/**
 * Purpose: Compute Struve function H1(x).
 * Source: https://dlmf.nist.gov/11.2#i
 * @param X x --- Argument of H1(x)
 * @param SH0 SH0 --- H0(x). The return value is written to the direction of the pointer.
 */
double ExcitonTB::STVH1(double X) const
{
    int N = 1000;

    double result = 0.0;
    double term_previous = 0.0;
    double term = 1.0;

    for (int n = 0; n <= N; ++n)
    {
        term = std::pow(-1,n)*std::pow(X/2.0, 2*n) / (std::tgamma(n + 1.5) * std::tgamma(n + 2.5));
        result += term;

        if (std::abs(term) < 1E-10)
        {
            break;
        } else if (n == N)
        {
            N = N + 10;
        }
        

    }

    return std::pow(0.5*X, 2)*result;
}

/** 
 * Calculate value of interaction potential (Keldysh). Units are eV.
 * @details If the distance is zero, then the interaction is renormalized to be V(a) since
 * V(0) is infinite, where a is the lattice parameter. Also, for r > cutoff the interaction is taken to be zero.
 * @param r Distance at which we evaluate the potential.
 * @return Value of Keldysh potential, V(r).
 */
double ExcitonTB::keldysh(double r) const {
    double eps_bar = (eps_m + eps_s)/2;
    double SH0;
    double cutoff = arma::norm(system->bravaisLattice.row(0)) * cutoff_ + 1E-5;
    double R = abs(r)/r0;
    double potential_value;
    if(r == 0){
        STVH0(regularization/r0, &SH0);
        potential_value = ec/(8E-10*eps0*eps_bar*r0)*(SH0 - y0(regularization/r0));
    }
    else if (r > cutoff){
        potential_value = 0.0;
    }
    else{
        STVH0(R, &SH0);
        potential_value = ec/(8E-10*eps0*eps_bar*r0)*(SH0 - y0(R));
    };

    return potential_value;
};

/**
 * Coulomb potential in real space.
 * @param r Distance at which we evaluate the potential.
 * @param regularization Regularization distance to remove divergence at r=0.
 * @return Value of Coulomb potential, V(r).
 */
double ExcitonTB::coulomb(double r) const {
    double cutoff = arma::norm(system->bravaisLattice.row(0)) * cutoff_ + 1E-5;
    r = abs(r);
    if (r > cutoff){
        return 0.0;
    }
    return (r != 0) ? ec/(4E-10*PI*eps0*r) : ec*1E10/(4*PI*eps0*regularization);    
}

/**
 * Coulomb potential in real space, with support for on-site energies (at the moment input parameters).
 * @param r Distance at which we evaluate the potential.
 * @param i Index of the atom, in case we are computing the Coulomb potential on-site (r=0)
 * @return Value of Coulomb potential, V(r).
 */
double ExcitonTB::coulomb(double r, uint i, uint j) const {
    double cutoff = arma::norm(system->bravaisLattice.row(0)) * cutoff_ + 1E-5;
    r = abs(r);

    if (r < 10E-5 && i == j) {

        if ((int)i + 1 > system->natoms) {
            std::cout << "Index out of bounds, must be <= natoms - 1. Exiting" << std::flush;
            exit(0);
        }

        if (system->motif.col(4)(i) == 0)
        {
            std::cout << "Something went wrong, here is the regularization value from input model: " << system->motif.col(4)(i) << std::endl;
        }
        
        return system->motif.col(4)(i);
    } else{
        return ec/(4E-10*PI*eps0*r);   
    }
}

/**
 * RPA potential in real space.
 * Implementation in progress, returns the same as keldysh.
 */
double ExcitonTB::rpa(double r) const {
    double eps_bar = (eps_m + eps_s)/2;
    double SH0;
    double cutoff = arma::norm(system->bravaisLattice.row(0)) * cutoff_ + 1E-5;
    double R = abs(r)/r0;
    double potential_value;
    if(r == 0){
        STVH0(regularization/r0, &SH0);
        potential_value = ec/(8E-10*eps0*eps_bar*r0)*(SH0 - y0(regularization/r0));
    }
    else if (r > cutoff){
        potential_value = 0.0;
    }
    else{
        STVH0(R, &SH0);
        potential_value = ec/(8E-10*eps0*eps_bar*r0)*(SH0 - y0(R));
    };

    return potential_value;
};

/**
 * Method to select the potential to be used in the of the exciton calculation.
 * @param potential Potential to be used in the direct term.
 * @return Pointer to function representing the potential.
 */

 const potptr ExcitonTB::selectPotential(std::string potential){
    if(potential == "keldysh"){
        return &ExcitonTB::keldysh;
    }
    else if(potential == "coulomb"){
        return &ExcitonTB::coulomb;
    }
    else if(potential == "rpa"){
        return &ExcitonTB::keldysh;
    }
    else{
        throw std::invalid_argument("selectPotential(): potential must be either 'keldysh', 'coulomb' or 'rpa'");
    }
}

/**
 * Method to select the potential to be used in the exciton calculation in reciprocal space.
 * @param potential Potential to be used in the direct term.
 * @return Pointer to function representing the potential.
 */
recpotptr ExcitonTB::selectReciprocalPotential(std::string potential){ //This function is not being used for now, as rpaFT returns a std::complex<double> and not double.
    if(potential == "keldysh"){
        return &ExcitonTB::keldyshFT;
    }
    else if(potential == "coulomb"){
        return &ExcitonTB::coulombFT;
    }
    else if(potential == "rpa"){
        return &ExcitonTB::keldyshFT;
    }
    else{
        throw std::invalid_argument("selectreciprocalPotential(): potential must be either 'keldysh', 'coulomb' or 'rpa'");
    }
}





/*---------------------------------------- Fourier transforms ----------------------------------------*/
/**
 * Evaluates the 2D Fourier transform of the Coulomb potential, which is an analytical expression.
 * @param k kpoint where we evaluate the FT.
 * @return 2D Fourier transform of the potential at q, FT[V](q).
 */
double ExcitonTB::coulomb_2D_FT(const arma::rowvec& k) const {

    double potential = 0;
    double eps = 1E-12;
    double knorm = arma::norm(k);
    if (knorm < eps){
        potential = 0;
    }
    else{
        potential = 1/knorm;
    }
    
    potential = potential*ec*1E10/(2*eps0*system->unitCellArea);
    
    return potential;
}  

/**
 * Evaluates the 2D Fourier transform of the Coulomb potential, which is an analytical expression.
 * @param g Index of G.
 * @param g2 Index of G2.
 * @param q kpoint where we evaluate the FT.
 * @return 2D Fourier transform of the potential at q, FT[V](q).
 */
double ExcitonTB::coulombFT(int g, int g2, const arma::rowvec q) const {

    if (g != g2){
        return 0.;
    }

    auto G = this->trunreciprocalLattice_.row(g);
    
    double potential = coulomb_2D_FT(q + G);
    
    return potential;
}  

/**
 * Evaluates the Fourier transform of the Keldysh potential, which is an analytical expression.
 * @param q kpoint where we evaluate the FT.
 * @return Fourier transform of the potential at q, FT[V](q).
 */
double ExcitonTB::keldyshFT(int g, int g2, arma::rowvec q) const {

    if (g != g2){
        return 0.;
    }

    double radius = cutoff*arma::norm(system->reciprocalLattice.row(0));
    double potential = 0;
    double eps_bar = (eps_m + eps_s)/2;
    double eps = arma::norm(system->reciprocalLattice.row(0))/totalCells;

    auto G = this->trunreciprocalLattice_.row(g);

    double qnorm = arma::norm(q + G);
    if (qnorm < eps){
        potential = 0;
        // double percentage = 0.48;
        // double q0 = percentage*arma::norm(arma::rowvec({0.024184, 0.0418879})); //2/r0;
        // potential = (2/q0 - r0); // Introduces regularization for Ryova-Keldysh potential in momentum space, Phys. Rev. B 88, 245309 (2013)
    }
    else{
        potential = 1/(qnorm*(1 + r0*qnorm));
    }
    
    potential = potential*ec*1E10/(2*eps0*eps_bar*system->unitCellArea);
    return potential;
}

/**
 * Evaluates the Fourier transform of the Keldysh potential, which is an analytical expression.
 * @param q kpoint where we evaluate the FT.
 * @return Fourier transform of the potential at q, FT[V](q).
 */
double ExcitonTB::keldyshFT(arma::rowvec q) const {

    double potential = 0;
    double eps_bar = (eps_m + eps_s)/2;
    double eps = 10E-12;

    double qnorm = arma::norm(q);

    if (qnorm < eps){
        return this->W00_at_0_;
    }
    else{
        potential = 1/(qnorm*(1 + r0*qnorm));
    }
    
    potential = potential*ec*1E10/(2*eps0*eps_bar*system->unitCellArea);

    return potential;
}


/**
 * Evaluates the RPA potential in reciprocal space.
 * @param q kpoint where we evaluate the FT.
 * @return W_(G,G')(q) matrix element of the screened potential
 */
std::complex<double> ExcitonTB::rpaFT(int g, int g2, arma::rowvec q) const {

    if (this->Invepsilonmatrix_.is_empty()){
        std::cerr << "Inverse of dielectric matrix has not been computed yet. Terminating." << std::endl;
    }

    std::complex<double> potential = 0;
    double eps = 1E-12; // arma::norm(system->reciprocalLattice.row(0))/totalCells;

    double qnorm = arma::norm(q);

    if (qnorm < eps && g == 0 && g2 == 0){
        potential = this->W00_at_0_;
    }
    else if ((qnorm < eps && g == 0 && g2 != 0) || (qnorm < eps && g != 0 && g2 == 0))
    {
        potential = 0.0;
    }
    else
    {
        int iq = system->findEquivalentPointBZ(q, this->ncell_);
        potential = std::sqrt(coulombFT(g, g, q)) * this->Invepsilonmatrix_.slice(iq).row(g)(g2) * std::sqrt(coulombFT(g2, g2, q));
    }

    return potential;
}

/**
 * Routine to compute the lattice Fourier transform with the potential displaced by some
 * vectors of the motif.
 * @param fAtomIndex Index of first atom of the motif.
 * @param sAtomIndex Index of second atom of the motif.
 * @param k kpoint where we evaluate the FT.
 * @param cells Matrix with the unit cells over which we sum to compute the lattice FT.
 * @param potential Pointer to potential function.
 * @return Motif lattice Fourier transform of the Keldysh potential at k.
 */
std::complex<double> ExcitonTB::motifFourierTransform(int fAtomIndex, int sAtomIndex, const arma::rowvec& k, 
                                                      const arma::mat& cells, potptr potential){

    std::complex<double> imag(0,1);
    std::complex<double> Vk = 0.0;
    arma::rowvec firstAtom = system->motif.row(fAtomIndex).subvec(0, 2);
    arma::rowvec secondAtom = system->motif.row(sAtomIndex).subvec(0, 2);

    for(uint n = 0; n < cells.n_rows; n++){
        arma::rowvec cell = cells.row(n);
        double module = arma::norm(cell + firstAtom - secondAtom);
        Vk += (this->*potential)(module)*std::exp(imag*arma::dot(k, cell));
    }
    Vk /= pow(totalCells, 1);

    return Vk;
}

/**
 * Method to compute the motif FT matrix at a given k vector.
 * @param k k vector where we compute the motif FT.
 * @param cells Matrix of unit cells over which the motif FT is computed.
 * @param potential Pointer to potential function.
 * @return void
 */
arma::cx_mat ExcitonTB::motifFTMatrix(const arma::rowvec& k, const arma::mat& cells, potptr potential){
    // Uses hermiticity of V
    int natoms = system->natoms;
    arma::cx_mat motifFT = arma::zeros<arma::cx_mat>(natoms, natoms);

    for(int fAtomIndex = 0; fAtomIndex < natoms; fAtomIndex++){
        for(int sAtomIndex = fAtomIndex; sAtomIndex < natoms; sAtomIndex++){
            motifFT(fAtomIndex, sAtomIndex) = motifFourierTransform(fAtomIndex, sAtomIndex, k, cells, potential);
            motifFT(sAtomIndex, fAtomIndex) = conj(motifFT(fAtomIndex, sAtomIndex));
        }   
    }

    return motifFT;
}

/**
 * Method to extend the motif Fourier transform matrix to match the dimension of the
 * one-particle basis. 
 * @param motifFT Matrix storing the motif Fourier transform to be extended.
 * @return Extended matrix.
 */
arma::cx_mat ExcitonTB::extendMotifFT(const arma::cx_mat& motifFT){
    arma::cx_mat extendedMFT = arma::zeros<arma::cx_mat>(system->basisdim, system->basisdim);
    int rowIterator = 0;
    int colIterator = 0;
    for(unsigned int atom_index_r = 0; atom_index_r < system->motif.n_rows; atom_index_r++){
        int species_r = system->motif.row(atom_index_r)(3);
        int norbitals_r = system->orbitals(species_r);
        colIterator = 0;
        for(unsigned int atom_index_c = 0; atom_index_c < system->motif.n_rows; atom_index_c++){
            int species_c = system->motif.row(atom_index_c)(3);
            int norbitals_c = system->orbitals(species_c);
            extendedMFT.submat(rowIterator, colIterator, 
                               rowIterator + norbitals_r - 1, colIterator + norbitals_c - 1) = 
                          motifFT(atom_index_r, atom_index_c) * arma::ones(norbitals_r, norbitals_c);
            colIterator += norbitals_c;
        }
        rowIterator += norbitals_r;
    }

    return extendedMFT;
}


/*------------------------------------ Interaction matrix elements ------------------------------------*/

/** 
 * Real space implementation of interaction term, valid for both direct and exchange.
 * To compute the direct term, the expected order is (ck,v'k',c'k',vk).
 * For the exchange term, the order is (ck,v'k',vk,c'k').
 * @param coefsK1 First eigenstate vector.
 * @param coefsK2 Second eigenstate vector.
 * @param coefsK3 Third eigenstate vector.
 * @param coefsK4 Fourth eigenstate vector.
 * @param motifFT Motif Fourier transform.
 * @return Interaction term. 
 */
std::complex<double> ExcitonTB::realSpaceInteractionTerm(const arma::cx_vec& coefsK1, 
                                     const arma::cx_vec& coefsK2,
                                     const arma::cx_vec& coefsK3, 
                                     const arma::cx_vec& coefsK4,
                                     const arma::cx_mat& motifFT){
    
    arma::cx_vec firstCoefArray =  arma::conj(coefsK1) % coefsK3;
    arma::cx_vec secondCoefArray = arma::conj(coefsK2) % coefsK4;
    /* Old implementation; one below should be faster */
    // std::complex<double> term = arma::dot(firstCoefArray, extendMotifFT(motifFT) * secondCoefArray);

    /* Instead of extending the motifFT matrix, reduce the coefs vectors */
    arma::cx_vec reducedFirstCoefArray = arma::zeros<arma::cx_vec>(system->natoms);
    arma::cx_vec reducedSecondCoefArray = arma::zeros<arma::cx_vec>(system->natoms);

    int iterator = 0;
    for(unsigned int atom_index = 0; atom_index < system->motif.n_rows; atom_index++){
        int norbitals = system->orbitals(system->motif.row(atom_index)(3));

        reducedFirstCoefArray(atom_index) = arma::sum(firstCoefArray.subvec(iterator, iterator + norbitals - 1));
        reducedSecondCoefArray(atom_index) = arma::sum(secondCoefArray.subvec(iterator, iterator + norbitals - 1));

        iterator += norbitals;
    }

    std::complex<double> term = arma::dot(reducedFirstCoefArray, motifFT * reducedSecondCoefArray);

    return term;
};

/**
 * Reciprocal space implementation of interaction term, valid for both direct and exchange.
 * @param coefsK Vector of eigenstate |v,k>.
 * @param coefsK2 Vector of eigenstate |v',k'>.
 * @param coefsKQ Vector of eigenstate |c,k+Q>.
 * @param coefsK2Q Vector of eigenstate |c',k'+Q>.
 * @param k kpoint corresponding to k.
 * @param k2 kpoint corresponding to k'.
 * @param kQ kpoint corresponding to k + Q.
 * @param k2Q kpoint corresponding to k' + Q.
 * @param potential Potential used to compute the matrix element (Coulomb, Keldysh or dielectric function within RPA)
 * @param nGs Number of reciprocal vectors included in the sum over G (and G')
 * @return Interaction term.
 */
std::complex<double> ExcitonTB::reciprocalInteractionTerm(const arma::cx_vec& coefsK, 
                                     const arma::cx_vec& coefsK2,
                                     const arma::cx_vec& coefsKQ, 
                                     const arma::cx_vec& coefsK2Q,
                                     const arma::rowvec& k, 
                                     const arma::rowvec& k2,
                                     const arma::rowvec& kQ, 
                                     const arma::rowvec& k2Q,
                                     std::string potential,
                                     int nGs) {
    std::complex<double> Ic, Iv;
    std::complex<double> term = 0;
    double radius = cutoff * arma::norm(system->reciprocalLattice.row(0));
    arma::mat reciprocalVectors = this->trunreciprocalLattice_;

    arma::rowvec null_vector(3,arma::fill::zeros);

    arma::rowvec k_dif = k - k2;
    // arma::rowvec kQ_dif = kQ - k2Q;

    int k_eff_index = system_->findEquivalentPointBZ(k_dif, ncell);
    // int kQ_eff_index = system_->findEquivalentPointBZ(kQ_dif, ncell);
    arma::rowvec k_eff = system_->kpoints.row(k_eff_index);
    // arma::rowvec kQ_eff = system->kpoints.row(kQ_eff_index);

    arma::rowvec g = k_dif - k_eff;

    uint g_index = 0;

    uint firstGvector = reciprocalVectors.n_rows > 1 ? 1 : 0;
    if (this->Gc_exciton_ < arma::norm(reciprocalVectors.row(firstGvector))){
        g_index = this->fetchReciprocalLatticeVector(g);
        g = null_vector;
    }

    if (potential == "coulomb"){ //There should be a better way of selecting the potential. Problem is rpaFT returns a std::complex<double>, and not double.
        for(int ig = 0; ig < nGs; ig++){
            arma::rowvec G = reciprocalVectors.row(ig);

            Ic = blochCoherenceFactor(coefsK2Q, coefsKQ, kQ, k2Q, G);
            Iv = blochCoherenceFactor(coefsK2, coefsK, k, k2, G);

            term += conj(Ic)*this->coulombFT(ig, ig, k - k2 + G)*Iv;
        }
    } else if (potential == "keldysh"){
        for(int ig = 0; ig < nGs; ig++){
            auto G = reciprocalVectors.row(ig);

            Ic = blochCoherenceFactor(coefsK2Q, coefsKQ, kQ, k2Q, G);
            Iv = blochCoherenceFactor(coefsK2, coefsK, k, k2, G);

            term += conj(Ic)*this->keldyshFT(k - k2 + G)*Iv;
        }
    } else if (potential == "rpa"){
        for(int ig = 0; ig < nGs; ig++){
            
            auto G = reciprocalVectors.row(ig) - g;
            
            for(int ig2 = 0; ig2 < nGs; ig2++){

                auto G2 = reciprocalVectors.row(ig2) - g;

                Ic = blochCoherenceFactor(coefsK2Q, coefsKQ, kQ, k2Q, G);
                Iv = blochCoherenceFactor(coefsK2, coefsK, k, k2, G2);

                term += conj(Ic)*this->rpaFT(ig + g_index, ig2 + g_index, k_eff)*Iv;
            }
        }
    } else {
        std::cout << "Potential not valid" << std::endl;
        exit(1);
    }
    
    // int g = 0;
    // int g2 = 0;
    // arma::rowvec G = reciprocalVectors.row(g);
    // arma::rowvec G2 = reciprocalVectors.row(g2);
    // while(arma::norm(G) <= radius){
    //     G = reciprocalVectors.row(g);
    //     while(arma::norm(G2) <= radius){
    //         G2 = reciprocalVectors.row(g2);
    //         Ic = blochCoherenceFactor(coefsKQ, coefsK2Q, kQ, k2Q, G);
    //         Iv = blochCoherenceFactor(coefsK, coefsK2, k, k2, G2);

    //         term += Ic*(this->*potential)(g, g2, k - k2)*conj(Iv);

    //         g2++;
    //     }
    //     g++;
    // }

    return term/((std::complex<double>)totalCells);
};

/**
 * Calculation of Bloch coherence factors, required to compute the interaction terms in reciprocal space.
 * @param coefs1 Vector of eigenstate |n,k>.
 * @param coefs2 Vector of eigenstate |n',k'>.
 * @param k1 kpoint k.
 * @param k2 kpoint k'.
 * @param G Reciprocal lattice vector used to compute the coherence factor.
 * @return Bloch coherence factor I = <n'k'| e^{-i(q + G).r} |nk> evaluated at G for states |nk>, |n'k'>.
 */
std::complex<double> ExcitonTB::blochCoherenceFactor(const arma::cx_vec& coefs1, const arma::cx_vec& coefs2,
                                                    const arma::rowvec& k1, const arma::rowvec& k2,
                                                    const arma::rowvec& G) const {

    std::complex<double> imag(0, 1);
    arma::cx_vec coefs = arma::conj(coefs1) % coefs2;
    arma::cx_vec phases = arma::ones<arma::cx_vec>(system->basisdim);

    int index_min = 0;
    int index_max = -1;

    for(int i = 0; i < system->natoms; i++){
        int species = system->motif.row(i)(3);
        arma::rowvec atomPosition = system->motif.row(i).subvec(0, 2); 

        index_max += system->orbitals(species);
        phases.subvec(index_min, index_max) *= exp(-imag*arma::dot(k1 - k2 + G, atomPosition));

        index_min = index_max + 1;
    }

    std::complex<double> factor = arma::dot(coefs, phases);

    return factor;
}


/**
 * Calculation of Bloch coherence factor taking into account the monolayer's finite thickness d
 * @param coefs1 Vector of eigenstate |n,k>.
 * @param coefs2 Vector of eigenstate |n',k'>.
 * @param k1 kpoint k.
 * @param k2 kpoint k'.
 * @param G Reciprocal lattice vector used to compute the coherence factor.
 * @param d Monolayer thickness.
 * @return Bloch coherence factor I = <n'k'| e^{-i(q + G).r} |nk> evaluated at G for states |nk>, |n'k'>, for finite thickness.
 */
std::complex<double> ExcitonTB::blochCoherenceFactor(const arma::cx_vec &coefs1, const arma::cx_vec &coefs2,
                                                     const arma::rowvec &k1, const arma::rowvec &k2,
                                                     const arma::rowvec &G, const double d) const {

    std::complex<double> imag(0, 1);
    arma::cx_vec coefs = arma::conj(coefs1) % coefs2;
    arma::cx_vec phases = arma::ones<arma::cx_vec>(system->basisdim);

    int index_min = 0;
    int index_max = -1;

    for (int i = 0; i < system->natoms; i++)
    {
        int species = system->motif.row(i)(3);
        arma::rowvec atomPosition = system->motif.row(i).subvec(0, 2);

        double extra_factor = std::sqrt(2*(1 - exp(-arma::norm(k1 - k2 + G) * d/2) * cosh(arma::norm(k1 - k2 + G) * atomPosition(2)))/(d*arma::norm(k1 - k2 + G)));

        index_max += system->orbitals(species);
        phases.subvec(index_min, index_max) *= extra_factor * exp(-imag * arma::dot(k1 - k2 + G, atomPosition));

        index_min = index_max + 1;
    }

    std::complex<double> factor = arma::dot(coefs, phases);

    return factor;
}

/*------------------------------------ Electron-hole pair basis ------------------------------------*/

/**
 * Method to generate a basis which is a subset of the basis considered for the
 * exciton. Its main purpose is to allow computation of Fermi golden rule between
 * two specified subbasis. 
 * @param bands Band subset of the originally specified for the exciton.
 * @return Matrix with the states corresponding to the specified subset.
 */
arma::imat ExcitonTB::specifyBasisSubset(const arma::ivec& bands){

    // Check if given bands vector corresponds to subset of bands
    try{
        for (const auto& band : bands){
            for (const auto& reference_band : bandList){
                if ((band + system->fermiLevel - reference_band) == 0) {
                    continue;
                }
            }
            throw "Error: Given band list must be a subset of the exciton one"; 
        };
    }
    catch (std::string e){
        std::cerr << e;
    };

    int reducedBasisDim = system->nk*bands.n_elem;
    std::vector<arma::s64> valence, conduction;
    for(uint i = 0; i < bands.n_elem; i++){
        if (bands(i) <= 0){
            valence.push_back(bands(i) + system->fermiLevel);
        }
        else{
            conduction.push_back(bands(i) + system->fermiLevel);
        }
    }
    arma::ivec valenceBands = arma::ivec(valence);
    arma::ivec conductionBands = arma::ivec(conduction);

    arma::imat states = createBasis(conductionBands, valenceBands);

    return states;
}

/**
 * Criterium to fix the phase of the single-particle eigenstates after diagonalization.
 * @details The prescription we take here is to impose that the sum of all the coefficients is real.
 * @return Fixed coefficients. 
 */
arma::cx_mat ExcitonTB::fixGlobalPhase(arma::cx_mat& coefs){

    arma::cx_rowvec sums = arma::sum(coefs);
    std::complex<double> imag(0, 1);
    for(arma::uword j = 0; j < sums.n_elem; j++){
        double phase = arg(sums(j));
        coefs.col(j) *= exp(-imag*phase);
    }

    return coefs;
}

/*------------------------------------ Initializers ------------------------------------*/

/**
 * Method to initialize the motif Fourier transform for all possible motif combination 
 * at a given kpoint.
 * @param i Index of kpoint.
 * @param cells Matrix of unit cells over which the motif FT is computed.
 * @param potential Pointer to potential function.
 * @return void
 */
void ExcitonTB::initializeMotifFT(int i, const arma::mat& cells, potptr potential){
    ftMotifStack.slice(i) = motifFTMatrix(system->meshBZ.row(i), cells, potential);
}


/**
 * Main method to compute all the relevant single-particle quantities (bands, eigenstates and fourier transforms),
 * to compute the Bethe-Salpeter equation.
 * @details It precomputes and saves the relevant data in the heap for later computations.
 * @param triangular Boolean to specify whether the Hamiltonian matrices are triangular (default = false).
 * @return void
 */ 
void ExcitonTB::initializeResultsH0(){

    int nTotalBands = bandList.n_elem;
    double radius = arma::norm(system->bravaisLattice.row(0)) * cutoff_;
    arma::mat cells = system_->truncateSupercell(ncell, radius);

    uint nk = system->nk;
    uint nk_aux = this->nk_aux;

    int natoms = system->natoms;
    int basisdim = system->basisdim;

    this->eigvecKStack_  = arma::cx_cube(basisdim, nTotalBands, nk);
    this->eigvecKQStack_ = arma::cx_cube(basisdim, nTotalBands, nk);
    this->eigvalKStack_  = arma::mat(nTotalBands, nk);
    this->eigvalKQStack_ = arma::mat(nTotalBands, nk);
    this->ftMotifStack   = arma::cx_cube(natoms, natoms, system->meshBZ.n_rows);
    this->ftMotifQ       = arma::cx_mat(natoms, natoms);

    vec auxEigVal(basisdim);
    arma::cx_mat auxEigvec(basisdim, basisdim);
    arma::cx_mat h;

    // Progress bar variables
    int step = 1;
	int displayNext = step;
	int percent = 0;

    system_->calculateInverseReciprocalMatrix();
    std::complex<double> imag(0, 1);

    std::cout << "Diagonalizing H0 for all k points... " << std::flush;
    for (uint i = 0; i < nk; i++){
        arma::rowvec k = system->kpoints.row(i);
        system->solveBands(k, auxEigVal, auxEigvec);

        auxEigvec = fixGlobalPhase(auxEigvec);
        eigvalKStack_.col(i) = auxEigVal(bandList);
        eigvecKStack_.slice(i) = auxEigvec.cols(bandList);

        if(arma::norm(Q) != 0){
            arma::rowvec kQ = system->kpoints.row(i) + Q;
            system->solveBands(kQ, auxEigVal, auxEigvec);

            auxEigvec = fixGlobalPhase(auxEigvec);
            eigvalKQStack_.col(i) = auxEigVal(bandList);
            eigvecKQStack_.slice(i) = auxEigvec.cols(bandList);
        }
        else{
            eigvecKQStack_.slice(i) = eigvecKStack.slice(i);
            eigvalKQStack_.col(i) = eigvalKStack.col(i);
        };
        
    };
    std::cout << "Done" << std::endl;

    if(this->mode == "realspace"){
        std::cout << "Computing lattice Fourier transform... " << std::flush;

        potptr directPotential = selectPotential(this->potential_);
        #pragma omp parallel for
        for (unsigned int i = 0; i < system->meshBZ.n_rows; i++){
            // BIGGEST BOTTLENECK OF THE CODE
            initializeMotifFT(i, cells, directPotential);     

                /* AJU 24-11-23: Progress bar does not work properly with parallel for 
                   Fix it or remove it direcly? */
                // percent = (100 * (i + 1)) / meshBZ.n_rows ;
                // if (percent >= displayNext){
                //    std::cout << "\r" << "[" << std::string(percent / 5, '|') << std::string(100 / 5 - percent / 5, ' ') << "]";
                //     std::cout << percent << "%";
                //     std::cout.flush();
                //     displayNext += step;
                // }
        }
        std::cout << "Done\n" << std::endl;
    }

    if (isscreeningset == true){
       
        std::cout << "Diagonalizing H0(k) for all k points in the coarser BZ mesh, and storing data in dielectric function's stack... " << std::flush;
        
        for (uint i = 0; i < nk_aux; i++){
            arma::rowvec k = this->kpoints_aux.row(i);
            system->solveBands(k, auxEigVal, auxEigvec);

            auxEigvec = fixGlobalPhase(auxEigvec);
            eigvalkStack_.col(i) = auxEigVal;
            eigveckStack_.slice(i) = auxEigvec;
        };
        std::cout << "Done" << std::endl;
    }

    if(this->exchange){
        potptr exchangePotential = selectPotential(this->exchangePotential_);
        this->ftMotifQ = motifFTMatrix(this->Q, cells, exchangePotential);
    }
};

/**
 * Routine to initialize the required variables to construct the Bethe-Salpeter Hamiltonian.
 * @param triangular Boolean to specify whether the single-particle Hamiltonian matrices are triangular.
 * @return void.
 */
void ExcitonTB::initializeHamiltonian(){

    if(bands.empty()){
        throw std::invalid_argument("Error: Exciton object must have some bands");
    }
    if(system->nk == 0){
        throw std::invalid_argument("Error: BZ mesh must be initialized first");
    }

    if (this->regularization_ < 1E-10){
        regularization_ = system->a;
    }

    this->excitonbasisdim_ = system->nk*valenceBands.n_elem*conductionBands.n_elem;
    this->totalCells_ = pow(ncell*system->factor, system->ndim);
    // system->bravaisLattice.print("Bravais Lattice vectors:");
    // system->motif.print("Motif vectors:");
    std::cout << "Initializing basis for BSE... " << std::flush;
    initializeBasis();
    generateBandDictionary();

    initializeResultsH0();
}

/*------------------------------------ Static dielectric function matrix elements ------------------------------------*/
/**
 * Method to compute the (G,G') matrix element of the static polarizability at the specified momentum vector q in the input file.
 * @details Writes the polarizability in the file polarizability_convergence.dat as a 
 * function of the number of included conduction bands. The polarizability is the purely 2D version.
 * @param q Momentum vector q specified in the input file
 * @return Polarizability
*/
std::complex<double> ExcitonTB::computesinglePolarizabilityMatrixElement(arma::rowvec &q, arma::rowvec &G, arma::rowvec& G2)
{
    std::ofstream polarfile("polarizability_convergence.dat"); 

    if (!polarfile.is_open()) { // check if the file was opened successfully
        std::cerr << "Error opening file\n";
    }

    uint nk = this->nk_aux;
    int basisdim = system->basisdim;
    arma::mat k_points = this->kpoints_aux;

    int nvbands = valencebands.size();
    int ncbands = conductionbands.size();

    int nvbandsincluded = this->nvalencebands_;
    int ncbandsincluded = this->nconductionbands_;

    int upperindexcband = nvbands + ncbandsincluded - 1;
    int lowerindexvbands = nvbands - nvbandsincluded;

    std::complex<double> g_s = this->g_s; // Spin degeneracy

    std::complex<double> term = 0.;
    std::complex<double> term_aux = 0.;
    std::complex<double> term_aux_2 = 0.;

    std::cout << "Done \n";
    
    arma::cx_vec coefskq, coefsk;
    arma::cx_vec coefskq_c, coefsk_v;


    for (int ic = nvbands; ic <= upperindexcband; ic++){
    
        for (int iv = nvbands - 1; iv >= nvbands - nvbandsincluded; iv--){

            for (uint ik = 0; ik < nk; ik++){

                arma::rowvec k = k_points.row(ik);
                arma::rowvec kq = k + q;
        
                // Using the atomic gauge
                if(gauge == "atomic"){
                    coefskq = system_->latticeToAtomicGauge(eigveckqStack_.slice(ik).col(iv), kq);
                    coefsk = system_->latticeToAtomicGauge(eigveckStack_.slice(ik).col(ic), k);
                
                    coefskq_c = system_->latticeToAtomicGauge(eigveckqStack_.slice(ik).col(ic), kq);
                    coefsk_v = system_->latticeToAtomicGauge(eigveckStack_.slice(ik).col(iv), k);
                }
                else{                            
                    coefskq = eigveckqStack_.slice(ik).col(iv);
                    coefsk = eigveckStack_.slice(ik).col(ic);

                    coefskq_c = eigveckqStack_.slice(ik).col(ic);
                    coefsk_v = eigveckStack_.slice(ik).col(iv);
                }

                std::complex<double> IvcG = blochCoherenceFactor(coefsk, coefskq, kq, k, G);
                std::complex<double> IvcG2 = blochCoherenceFactor(coefsk, coefskq, kq, k, G2);

                term += IvcG*std::conj(IvcG2) / (eigvalkqStack_.col(ik)(iv) - eigvalkStack_.col(ik)(ic));
            
                std::complex<double> IcvG = blochCoherenceFactor(coefsk_v, coefskq_c, kq, k, G);
                std::complex<double> IcvG2 = blochCoherenceFactor(coefsk_v, coefskq_c, kq, k, G2);
       
                term_aux += (IvcG*std::conj(IvcG2) / (eigvalkqStack_.col(ik)(iv) - eigvalkStack_.col(ik)(ic))) - (IcvG*std::conj(IcvG2) / (eigvalkqStack_.col(ik)(ic) - eigvalkStack_.col(ik)(iv)));
            }

            polarfile << nvbands - iv << " " << ic - nvbands + 1 << " " << real(g_s) * real(term_aux) / (nk) << " " << real(g_s) * imag(term_aux) / (nk) << "\n";
        }
    }

    polarfile.close();

    std::cout << "Chi = " << g_s*term_aux/((std::complex<double>)nk) << std::endl;

    return g_s*term_aux/((std::complex<double>)nk);
}

/**
 * Method to compute the (G,G') matrix element of the static 2D polarizability at the specified arbitrary momentum vector q.
 * @details The purely 2D polarizability is computed
 * @param G Reciprocal lattice vector G
 * @param G2 Reciprocal lattice vector G2
 * @param q Momentum vector q
 * @return Polarizability
*/
inline std::complex<double> ExcitonTB::compute_2D_PolarizabilityMatrixElement(const arma::rowvec& G, const arma::rowvec& G2, const arma::rowvec& q) {

    uint nk = this->nk_aux;
    int basisdim = system->basisdim;

    int nvbands = valencebands.size();
    int ncbands = conductionbands.size();

    int nvbandsincluded = this->nvalencebands_;
    int ncbandsincluded = this->nconductionbands_;

    int upperindexcband = nvbands + ncbandsincluded - 1;
    int lowerindexvbands = nvbands - nvbandsincluded;

    //int basisdim = nvbands + ncbands;

    arma::cx_vec coefskq, coefsk;
    arma::cx_vec coefskq_c, coefsk_v;

    std::complex<double> term = 0.;
    std::complex<double> term_aux = 0.;
    std::complex<double> term_aux_2 = 0.;
    std::complex<double> g_s = this->g_s; // Spin degeneracy

    if (this->eigveckqStack_.is_empty() && this->eigvalkqStack_.is_empty()) {
        this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
        this->eigvalkqStack_ = arma::mat(basisdim, nk);

        std::cout << "Bloch Hamiltonian has to be diagonalized at every k+q point before computing polarizability matrix elements. Terminating" << std::endl;
        exit(0);
    }

    for (int ic = nvbands; ic <= upperindexcband; ic++){
    
        for (int iv = nvbands - 1; iv >= nvbands - nvbandsincluded; iv--){

            for (uint ik = 0; ik < nk; ik++){

                arma::rowvec k = this->kpoints_aux.row(ik);
                arma::rowvec kq = k + q;
        
               // Using the atomic gauge
                if(gauge == "atomic"){
                    coefskq = system_->latticeToAtomicGauge(eigveckqStack_.slice(ik).col(iv), kq);
                    coefsk = system_->latticeToAtomicGauge(eigveckStack_.slice(ik).col(ic), k);
                
                    coefskq_c = system_->latticeToAtomicGauge(eigveckqStack_.slice(ik).col(ic), kq);
                    coefsk_v = system_->latticeToAtomicGauge(eigveckStack_.slice(ik).col(iv), k);
                }
                else{                            
                    coefskq = eigveckqStack_.slice(ik).col(iv);
                    coefsk = eigveckStack_.slice(ik).col(ic);

                    coefskq_c = eigveckqStack_.slice(ik).col(ic);
                    coefsk_v = eigveckStack_.slice(ik).col(iv);
                }

                std::complex<double> IvcG = blochCoherenceFactor(coefsk, coefskq, kq, k, G);
                std::complex<double> IvcG2 = blochCoherenceFactor(coefsk, coefskq, kq, k, G2);

                std::complex<double> IcvG = blochCoherenceFactor(coefsk_v, coefskq_c, kq, k, G);
                std::complex<double> IcvG2 = blochCoherenceFactor(coefsk_v, coefskq_c, kq, k, G2);

                term += IvcG*std::conj(IvcG2) / (eigvalkqStack_.col(ik)(iv) - eigvalkStack_.col(ik)(ic));

                // term_aux += IvcG*std::conj(IvcG2) / (eigvalkqStack_.col(ik)(iv) - eigvalkStack_.col(ik)(ic));
                // term_aux_2 += IcvG*std::conj(IcvG2)  / (eigvalkStack_.col(ik)(iv) - eigvalkqStack_.col(ik)(ic));
            }
        }
    }

    return g_s*2.0*term/((std::complex<double>)nk); // Factor of 2 from TRS
}

/**
 * Method to compute the (G,G') matrix element of the static Q2D polarizability at the specified momentum q.
 * @details The Q2D polarizability computed has meaning only when multiplied by the Coulomb potential
 * @param G Reciprocal lattice vector G
 * @param G2 Reciprocal lattice vector G2
 * @param q Momentum vector q
 * @param d Thickness of the 2D system
 * @return Polarizability
*/
inline std::complex<double> ExcitonTB::compute_quasi2D_PolarizabilityMatrixElement(const arma::rowvec& G, const arma::rowvec& G2, const arma::rowvec& q, double d) {

    uint nk = this->nk_aux;
    int basisdim = system->basisdim;

    int nvbands = valencebands.size();
    int ncbands = conductionbands.size();

    int nvbandsincluded = this->nvalencebands_;
    int ncbandsincluded = this->nconductionbands_;

    int upperindexcband = nvbands + ncbandsincluded - 1;
    int lowerindexvbands = nvbands - nvbandsincluded;

    //int basisdim = nvbands + ncbands;

    arma::cx_vec coefskq, coefsk;
    arma::cx_vec coefskq_c, coefsk_v;

    std::complex<double> term_aux = 0.;
    std::complex<double> term_aux_2 = 0.;    
    std::complex<double> g_s = this->g_s; // Spin degeneracy

    if (arma::norm(q) < 1E-7 && (arma::norm(G) < 1E-7 || arma::norm(G2) < 1E-7)){
        return 0.;
    }

    if (this->eigveckqStack_.is_empty() && this->eigvalkqStack_.is_empty()) {
        this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
        this->eigvalkqStack_ = arma::mat(basisdim, nk);

        std::cout << "Bloch Hamiltonian has to be diagonalized at every k+q point before computing polarizability matrix elements. Terminating" << std::endl;
        exit(0);
    }

    for (int ic = nvbands; ic <= upperindexcband; ic++){
    
        for (int iv = nvbands - 1; iv >= nvbands - nvbandsincluded; iv--){

            for (uint ik = 0; ik < nk; ik++){

                arma::rowvec k = this->kpoints_aux.row(ik);
                arma::rowvec kq = k + q;
        
               // Using the atomic gauge
                if(gauge == "atomic"){
                    coefskq = system_->latticeToAtomicGauge(eigveckqStack_.slice(ik).col(iv), kq);
                    coefsk = system_->latticeToAtomicGauge(eigveckStack_.slice(ik).col(ic), k);
                
                    coefskq_c = system_->latticeToAtomicGauge(eigveckqStack_.slice(ik).col(ic), kq);
                    coefsk_v = system_->latticeToAtomicGauge(eigveckStack_.slice(ik).col(iv), k);
                }
                else{                            
                    coefskq = eigveckqStack_.slice(ik).col(iv);
                    coefsk = eigveckStack_.slice(ik).col(ic);

                    coefskq_c = eigveckqStack_.slice(ik).col(ic);
                    coefsk_v = eigveckStack_.slice(ik).col(iv);
                }

                std::complex<double> IvcG_d = blochCoherenceFactor(coefsk_v, coefskq_c, kq, k, G, d);
                std::complex<double> IcvG_d = blochCoherenceFactor(coefsk, coefskq, kq, k, G, d);
               
                std::complex<double> IvcG2_d = blochCoherenceFactor(coefsk_v, coefskq_c, kq, k, G2, d);
                std::complex<double> IcvG2_d = blochCoherenceFactor(coefsk, coefskq, kq, k, G2, d);


                term_aux += IvcG_d*std::conj(IvcG2_d) / (eigvalkStack_.col(ik)(iv) - eigvalkqStack_.col(ik)(ic));
                
                term_aux_2 += IcvG_d*std::conj(IcvG2_d)  / (eigvalkStack_.col(ik)(ic) - eigvalkqStack_.col(ik)(iv));
            }
        }
    }

    return g_s*(term_aux - term_aux_2)/((std::complex<double>)nk);
}

/**
 * Method to compute the (G,G') matrix element of the static 2D dielectric matrix at the specified arbitrary momentum vector q.
 * @details The purely 2D polarizability is computed
 * @param G Reciprocal lattice vector G
 * @param G2 Reciprocal lattice vector G2
 * @param q Momentum vector q
 * @return Polarizability
*/
std::complex<double> ExcitonTB::compute_2D_DielectricMatrixElement(const arma::rowvec& G, const arma::rowvec& G2, const arma::rowvec& q) {

    double kroneckerdelta = arma::norm(G - G2) < 10E-7 ? 1 : 0;

    const double potential = std::sqrt(coulomb_2D_FT(G + q))*std::sqrt(coulomb_2D_FT(G2 + q));

    std::complex<double> epsilon = kroneckerdelta - potential*this->compute_2D_PolarizabilityMatrixElement(G,G2,q);
            
    return epsilon;
}

/**
 * Method to compute the (G,G') matrix element of the quasi-2D dielectric matrix at the specified arbitrary momentum vector q (divided by d).
 * @details The dielectric matrix as function of (z,z') averaged over the thickness d of the material is computed
 * @param G Reciprocal lattice vector G
 * @param G2 Reciprocal lattice vector G2
 * @param q Momentum vector q
 * @param d 2D material thickness
 * @return Polarizability
 */
std::complex<double> ExcitonTB::compute_quasi2D_DielectricMatrixElement(const arma::rowvec &G, const arma::rowvec &G2, const arma::rowvec &q, const double d) {
    
    double potential = std::sqrt(coulomb_2D_FT(q + G))*std::sqrt(coulomb_2D_FT(q + G2));

    double kroneckerdelta = arma::norm(G - G2) < 10E-4 ? 1 : 0;

    std::complex<double> Chi_Q2D = compute_quasi2D_PolarizabilityMatrixElement(G, G2, q, d);

    std::complex<double> epsilon = kroneckerdelta - potential*Chi_Q2D;

    return epsilon;
}

/**
 * Method to compute the (G,G') matrix element of the static 2D polarizability at the specified momentum vector q.
 * @details The momentum q has to be specified through an index, matching a k point in the BZ mesh.  The purely 2D polarizability is computed.
 * @param G Reciprocal lattice vector G
 * @param G2 Reciprocal lattice vector G2
 * @param iq Index of momentum vector q
 * @return Polarizability
*/
inline std::complex<double> ExcitonTB::compute_2D_PolarizabilityMatrixElement(const arma::rowvec& G, const arma::rowvec& G2, const int iq) {

    int nk = this->nk_aux;
    int natoms = system->natoms;
    int basisdim = system->basisdim;

    int nvbands = valencebands.size();
    int ncbands = conductionbands.size();

    int nvbandsincluded = this->nvalencebands_;
    int ncbandsincluded = this->nconductionbands_;

    int upperindexcband = nvbands + ncbandsincluded - 1;
    int lowerindexvbands = nvbands - nvbandsincluded;

    arma::cx_vec coefskq, coefsk;
    arma::cx_vec coefskq_c, coefsk_v;

    std::complex<double> term = 0.;
    std::complex<double> term_aux = 0.;
    std::complex<double> g_s = this->g_s; // Spin degeneracy

    for (int ik = 0; ik < nk; ik++){

        arma::rowvec k = this->kpoints_aux.row(ik);
        arma::rowvec kq = this->kpoints_aux.row(ik) + system->kpoints.row(iq);

        // int kqindex = system_->findEquivalentPointBZ(kq, ncell);

        const arma::cx_dmat *auxk_slice = &eigveckStack_.slice(ik);
        const arma::cx_dmat *auxkq_slice = &eigveckqStack_.slice(ik);

        for (int ic = nvbands; ic <= upperindexcband; ic++){

            // Using the atomic gauge
            if (gauge == "atomic"){
                coefsk = system_->latticeToAtomicGauge(auxk_slice->col(ic), k);
            } else {
                coefsk = auxk_slice->col(ic);
            }

            if (gauge == "atomic") {
                coefskq_c = system_->latticeToAtomicGauge(auxkq_slice->col(ic), kq);
            } else {
                coefskq_c = auxkq_slice->col(ic);
            }

            for (int iv = nvbands - 1; iv >= nvbands - nvbandsincluded; iv--){

                // Using the atomic gauge
                if (gauge == "atomic"){
                    coefskq = system_->latticeToAtomicGauge(auxkq_slice->col(iv), kq);
                } else {
                    coefskq = auxkq_slice->col(iv);
                }

                if (gauge == "atomic") {
                    coefsk_v = system_->latticeToAtomicGauge(auxk_slice->col(iv), k);
                } else {
                    coefsk_v = auxk_slice->col(iv);
                }

                std::complex<double> IvcG = blochCoherenceFactor(coefskq, coefsk, kq, k, G);
                std::complex<double> IvcG2 = blochCoherenceFactor(coefskq, coefsk, kq, k, G2);

                // term += IvcG * std::conj(IvcG2) / (eigvalkqStack_.col(ik)(iv) - eigvalkStack_.col(ik)(ic));

                std::complex<double> IcvG = blochCoherenceFactor(coefskq_c, coefsk_v, kq, k, G);
                std::complex<double> IcvG2 = blochCoherenceFactor(coefskq_c, coefsk_v, kq, k, G2);

                term_aux += std::conj(IvcG) * IvcG2 / (eigvalkqStack_.col(ik)(iv) - eigvalkStack_.col(ik)(ic)) - std::conj(IcvG) * IcvG2 / (eigvalkqStack_.col(ik)(ic) - eigvalkStack_.col(ik)(iv));
            }
        }
    }

    return g_s*term_aux/((std::complex<double>)totalCells);
}

/**
 * Method to compute the (G,G') matrix element of the static polarizability at the specified momentum vector q from list of q points read from file.
 * @details The momentum q has to be specified through the vector coordinates and index.  The purely 2D polarizability is computed.
 * @param G Reciprocal lattice vector G
 * @param G2 Reciprocal lattice vector G2
 * @param q Momentum vector q
 * @param iq Index of momentum vector q
 * @return Polarizability
 */
std::complex<double> ExcitonTB::compute_2D_PolarizabilityMatrixElement(const arma::rowvec &G, const arma::rowvec &G2, const arma::rowvec &q, const int iq)
{

    int nk = this->nk_aux;
    int natoms = system->natoms;
    int basisdim = system->basisdim;

    int nvbands = valencebands.size();
    int ncbands = conductionbands.size();

    int nvbandsincluded = this->nvalencebands_;
    int ncbandsincluded = this->nconductionbands_;

    int upperindexcband = nvbands + ncbandsincluded - 1;
    int lowerindexvbands = nvbands - nvbandsincluded;

    arma::cx_vec coefskq, coefsk;
    // arma::cx_vec coefskq_c, coefsk_v;

    std::complex<double> term = 0.;
    std::complex<double> g_s = this->g_s; // Spin degeneracy

    for (int ik = 0; ik < nk; ik++)
    {

        arma::rowvec k = this->kpoints_aux.row(ik);
        arma::rowvec kq = this->kpoints_aux.row(ik) + q;

        arma::cx_mat *auxk_slice = &eigveckStack_.slice(ik);
        //arma::cx_mat *auxkq_slice = &eigveckqStack_test[iq].slice(ik);

        for (int ic = nvbands; ic <= upperindexcband; ic++)
        {

            // Using the atomic gauge
            if (gauge == "atomic")
            {
                coefsk = system_->latticeToAtomicGauge(auxk_slice->col(ic), k);
            }
            else
            {
                coefsk = auxk_slice->col(ic);
            }

            // if (gauge == "atomic")
            // {
            //     coefskq_c = system_->latticeToAtomicGauge(eigveckqStack_.slice(ik).col(ic), q);
            // }
            // else
            // {
            //     coefskq_c = eigveckqStack_.slice(ik).col(ic);
            // }

            for (int iv = nvbands - 1; iv >= nvbands - nvbandsincluded; iv--)
            {

                // Using the atomic gauge
                if (gauge == "atomic")
                {
                    coefskq = system_->latticeToAtomicGauge(eigveckqStack_.slice(ik).col(iv), kq);
                }
                else
                {
                    coefskq = eigveckqStack_.slice(ik).col(iv);
                }

                // if (gauge == "atomic")
                // {
                //     coefsk_v = system_->latticeToAtomicGauge(auxk_slice->col(iv), k);
                // }
                // else
                // {
                //     coefsk_v = auxk_slice->col(iv);
                // }

                std::complex<double> IvcG = blochCoherenceFactor(coefskq, coefsk, kq, k, G);
                std::complex<double> IvcG2 = blochCoherenceFactor(coefskq, coefsk, kq, k, G2);

                //term += IvcG * std::conj(IvcG2) / (eigvalkStack_.col(kqindex)(iv) - eigvalkStack_.col(ik)(ic));

                // std::complex<double> IcvG = blochCoherenceFactor(coefskq_c, coefsk_v, kq, k, G);
                // std::complex<double> IcvG2 = blochCoherenceFactor(coefskq_c, coefsk_v, kq, k, G2);
                // term += std::conj(IvcG) * IvcG2 / (eigvalkqStack_test.slice(iq).col(ik)(iv) - eigvalkStack_.col(ik)(ic)) - std::conj(IcvG) * IcvG2 / (eigvalkqStack_test.slice(iq).col(ik)(ic) - eigvalkStack_.col(ik)(iv));
                term += std::conj(IvcG) * IvcG2 / (eigvalkqStack_test.slice(iq).col(ik)(iv) - eigvalkStack_.col(ik)(ic));
            }
        }
    }

    return g_s*2.0*term/((std::complex<double>)nk); // Factor of 2 from TRS
}

/**
 * Method to compute the (G,G') matrix element of the static polarizability at the specified arbitrary momentum vector q.
 * @details The momentum q has to be specified through an index, matching a k point in the BZ mesh. The purely 2D dielectric function is computed.
 * @param G Reciprocal lattice vector G
 * @param G2 Reciprocal lattice vector G2
 * @param iq Index of momentum vector q
 * @return Polarizability 
*/
std::complex<double> ExcitonTB::compute_2D_DielectricMatrixElement(const arma::rowvec& G, const arma::rowvec& G2, const int iq) {

    double kroneckerdelta = arma::norm(G - G2) < 10E-7 ? 1 : 0;

    double potential = coulomb_2D_FT(G + q);

    std::complex<double> epsilon = kroneckerdelta - potential*this->compute_2D_PolarizabilityMatrixElement(G,G2,iq);
            
    return epsilon;
}

/**
 * Method to compute the (G,G') static 2D polarizability in the BZ mesh.
 * @details Opens 'polarizability_mesh.dat' file and writes in it the values of the polarizability at each point in the BZ mesh
 * @return void
*/
void ExcitonTB::PolarizabilityMesh() {

    auto start = high_resolution_clock::now();

    if (this->mode == "realspace"){
        std::cout << "Numerical screening in real space not available yet. Terminating." << std::flush;
        exit(0);
    }

    if (this->mode == "reciprocalspace"){
        std::cout << "Computing polarizability in the BZ mesh... \n" << std::flush;

        std::ofstream polarfile; 

        polarfile.open("polarizability_mesh.dat");

        if (!polarfile.is_open()) { // check if the file was opened successfully
            std::cerr << "Error opening file\n";
            std::cerr << errno << "\n";
            
            exit(1);
        }

        uint nq = system->nk;
        uint nk_aux = this->nk_aux;
        int basisdim = system->basisdim;

        arma::cx_vec Chi(nq, arma::fill::zeros);

        double radius = cutoff * arma::norm(system->reciprocalLattice.row(0));
        arma::mat reciprocalVectors = this->trunreciprocalLattice_;

        arma::rowvec g = reciprocalVectors.row(this->Gs(0)); // Sets G
        arma::rowvec g2 = reciprocalVectors.row(this->Gs(1)); // Sets G'

        if (this->eigveckqStack_.is_empty() && this->eigvalkqStack_.is_empty()) {
            this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk_aux);
            this->eigvalkqStack_ = arma::mat(basisdim, nk_aux);
        }
        std::cout << "iq = ";
        // #pragma omp parallel for
        for (uint iq = 0; iq < nq; iq++){

            vec auxEigVal(basisdim);
            arma::cx_mat auxEigvec(basisdim, basisdim);

            arma::rowvec q = system->kpoints.row(iq);

            for (uint i = 0; i < nk_aux; i++)
            {
                arma::rowvec kq = this->kpoints_aux.row(i) + q;
                system->solveBands(kq, auxEigVal, auxEigvec);
                auxEigvec = fixGlobalPhase(auxEigvec);
                this->eigvalkqStack_.col(i) = auxEigVal;
                this->eigveckqStack_.slice(i) = auxEigvec;
            };
            
            std::cout << iq << " " << std::flush;
            
            Chi(iq) = this->compute_2D_PolarizabilityMatrixElement(g, g2, q);
        }

        std::cout << "\n Chi computed" << std::endl;
        
        for (uint iq = 0; iq < nq; iq++){
            auto q = system_->kpoints.row(iq);
            polarfile << q(0) << " " << q(1) << " " << q(2) << " " << real(Chi(iq)) << " " << imag(Chi(iq)) << "\n";
        }

        polarfile.close();
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Done in " << duration.count()/1000.0 << " s." << std::endl;
}

/**
 * Method to fetch the index of a reciprocal lattice vector.
 * @details returns -1 if not found.
 * @return int
*/
int ExcitonTB::fetchReciprocalLatticeVector(arma::rowvec G){

    uint nGs = this->trunreciprocalLattice_.n_rows;

	for (uint i = 0; i < nGs; ++i){
        if (arma::norm(G - this->trunreciprocalLattice_.row(i)) < 1E-7){
            return i;
        }
    }

    std::cout << "Reciprocal lattice vector G = (" << G(0) << ", " << G(1) << ") not found in the reciprocal lattice." << std::endl;

    return -1; // G not found
}

/**
 * Method to compute the static 2D dielectric matrix in the BZ mesh.
 * @return void
*/
void ExcitonTB::compute_2D_DielectricMatrix(){

    auto start = high_resolution_clock::now();

    if (this->mode == "realspace"){
        std::cout << "Implementation of the exciton in real space with numerical screening not available yet. Terminating" << std::endl;
        exit(0);
    }
    
    if (this->mode == "reciprocalspace"){

        arma::mat ReciprocalVectors = this->trunreciprocalLattice_;
        uint nGs = ReciprocalVectors.n_rows;
        uint Ncells = this->ncell_;
        uint odd = Ncells % 2;
        uint nq = odd == 1 ? Ncells * (Ncells - odd) / 2 + (Ncells - odd) / 2 + 1 : (2 * Ncells - 1) + (Ncells - 1) * (Ncells - 2) / 2 + Ncells / 2 + 1; // Only half of the BZ

        std::cout << "Number of G vectors included in the calculation: " << nGs << std::endl;
        printReciprocalLattice();

        this->setq_points_list(system->kpoints); // The set of q points where the dielectric matrix is computed coincides with the BZ mesh
        uint Nqpoints = this->qpoints_list_.n_rows;
        uint nk = Nqpoints;
        arma::mat q_points(Nqpoints, 3, arma::fill::zeros);

        uint ncell_aux = this->ncell_aux;
        uint nk_aux = this->nk_aux;
        uint Nktotal = system->nk;

        q_points = this->qpoints_list_; 
        
        uint basisdim = system->basisdim;
        
        std::cout << "Diagonalizing H(k+q) for every point q, for every point k... " << std::flush;

        // In case the polarizability/dielectric matrix have been computed before with another routine, reshape to account for a different number of q points
        if (this->eigvalkqStack_test.is_empty() || this->eigveckqStack_test.empty())
        {
            this->eigvalkqStack_test = arma::cube(basisdim, nk_aux, Nqpoints, arma::fill::zeros);

            std::vector<arma::cx_cube> vector_aux;
            vector_aux.resize(Nqpoints);
            this->eigveckqStack_test = vector_aux;

            for (uint iq = 0; iq < Nqpoints; ++iq)
            {
                this->eigveckqStack_test[iq] = arma::cx_cube(basisdim, basisdim, nk_aux, arma::fill::zeros);
            }
        }
        else
        {
            this->eigvalkqStack_test.set_size(basisdim, nk_aux, Nqpoints);
            this->eigveckqStack_test.resize(Nqpoints);

            for (uint iq = 0; iq < Nqpoints; ++iq)
            {
                this->eigveckqStack_test[iq] = arma::cx_cube(basisdim, basisdim, nk_aux, arma::fill::zeros);
            }
        }

        arma::vec auxEigVal(basisdim);
        arma::cx_mat auxEigvec(basisdim, basisdim);
        
        std::cout << "q point " << std::flush;

        for (uint iq = 0; iq < Nqpoints; ++iq)
        {
            arma::rowvec q = q_points.row(iq);

            for (uint i = 0; i < nk_aux; i++)
            {
                arma::rowvec kq = this->kpoints_aux.row(i) + q;
                system->solveBands(kq, auxEigVal, auxEigvec);

                auxEigvec = fixGlobalPhase(auxEigvec);
                this->eigvalkqStack_test.slice(iq).col(i) = auxEigVal;
                this->eigveckqStack_test[iq].slice(i) = auxEigvec;
            };

            if ((iq + 1) % 10 == 0) {
                std::cout << iq + 1 << ", " << std::flush;
            }
        }

        std::cout << "\nDone. \nComputing dielectric matrix in the BZ mesh... \n" << std::flush;

        arma::cx_mat auxvecsol(nGs,nGs,arma::fill::zeros);
        arma::cx_mat auxvec(nGs,nGs,arma::fill::eye);

        arma::imat indecesqg(nq*nGs*(nGs+1)/2,3,arma::fill::zeros);

        if (odd == 0){ // If Ncells is even, then q points at the boundaries of the BZ with no symmetric counterpart are computed seperately
            int i=0;
            
            for (uint iq = 0; iq < Ncells; iq++){ // 1st BZ boundary
                for (uint g = 0; g < nGs; g++){
                    for (uint g2 = g; g2 < nGs; g2++){
                        indecesqg.row(i)(0) = iq;
                        indecesqg.row(i)(1) = g;
                        indecesqg.row(i)(2) = g2;
                        i++;
                    }
                }
            }

            for (uint iq = 1; iq < Ncells; iq++){ // 2nd BZ boundary
                for (uint g = 0; g < nGs; g++){
                    for (uint g2 = g; g2 < nGs; g2++){
                        indecesqg.row(i)(0) = iq*Ncells;
                        indecesqg.row(i)(1) = g;
                        indecesqg.row(i)(2) = g2;
                        i++;
                    }
                }
            }

            for (uint iq = 1; iq < Ncells/2; iq++){ // Center-symmetric submesh of the full BZ mesh
                for (uint iq2 = 1; iq2 < Ncells; iq2++){    
                    for (uint g = 0; g < nGs; g++){
                        for (uint g2 = g; g2 < nGs; g2++){
                            indecesqg.row(i)(0) = iq2 + iq*Ncells;
                            indecesqg.row(i)(1) = g;
                            indecesqg.row(i)(2) = g2;
                            i++;
                        }
                    }
                }
            }

            for (uint iq = 0; iq < Ncells/2; iq++){ // Center-symmetric submesh of the full BZ mesh
                for (uint g = 0; g < nGs; g++){
                    for (uint g2 = g; g2 < nGs; g2++){
                        indecesqg.row(i)(0) = 1 + iq + Ncells*Ncells/2;
                        indecesqg.row(i)(1) = g;
                        indecesqg.row(i)(2) = g2;
                        i++;
                    }
                }
            }
        } else if (odd == 1) { // If Ncells is odd, the full BZ mesh is automatically center-symmetric
            int i=0;
            
            for (uint iq = 0; iq < nq; iq++){
                for (uint g = 0; g < nGs; g++){
                    for (uint g2 = g; g2 < nGs; g2++){
                        indecesqg.row(i)(0) = iq;
                        indecesqg.row(i)(1) = g;
                        indecesqg.row(i)(2) = g2;
                        i++;
                    }
                }
            }
        }

        std::cout << "Calculation of dielectric matrix at " << std::endl;
        if (odd == 0){

            std::cout << " (Number of k points is even. Computing the dielectric matrix at the nonsymmetric part of the BZ... )" << std::endl;
            uint iq_test = 0;
            arma::rowvec q_test = q_points.row(iq_test);

            double percentage = 0;
            double percentage_aux = 0;
            
            for (uint ik = 0; ik < (2 * Ncells - 1); ik++)
            {
                uint iq = indecesqg.row(ik*nGs*(nGs+1)/2)(0);
                arma::rowvec q = q_points.row(iq);

                this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
                this->eigveckqStack_ = this->eigveckqStack_test[iq];

                #pragma omp parallel for
                for (uint ig = 0; ig < nGs * (nGs + 1) / 2; ig++)
                {
                    uint g  = indecesqg.row(ig)(1);
                    uint g2 = indecesqg.row(ig)(2);

                    arma::rowvec G = ReciprocalVectors.row(g);
                    arma::rowvec G2 = ReciprocalVectors.row(g2);

                    std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(G, G2, q);

                    double potentialg =  std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q)); // coulombFT(g, g, q);
                    double potentialg2 = std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q)); // std::sqrt(coulombFT(g2, g2, q));

                    this->Chimatrix_.slice(iq).row(g)(g2) = Chi;
                    this->Chimatrix_.slice(iq).row(g2)(g) = std::conj(Chi);

                    double kroneckerdelta = g == g2 ? 1 : 0;

                    this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg * Chi;
                    this->epsilonmatrix_.slice(iq).row(g2)(g) = kroneckerdelta - potentialg2 * std::conj(Chi);
                }

                percentage = ((double)ik + 1)*100./(double)Nqpoints;

                if ((int) percentage % 5 == 0 && (int) percentage_aux != (int) percentage)
                {
                    std::cout << (int) percentage << "%%, " << std::flush;
                    percentage_aux = percentage;
                }
            }

            // Computes the dielectric matrix at the centrosymmetric submesh of the BZ

            std::cout << " (Computing the dielectric matrix at the centrosymmetric part of the BZ... )" << std::endl;

            uint iq_i = (2 * Ncells - 1);
            uint ik_f = (Ncells - 1)*(Ncells - 2)/2 + Ncells/2;
            for (uint ik = 0; ik < ik_f; ik++)
            {
                uint iq = indecesqg.row(iq_i*nGs*(nGs + 1)/2 + ik*nGs*(nGs + 1)/2)(0); // indecesqg.row(i_aux)(0);
                arma::rowvec q = q_points.row(iq);

                this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
                this->eigveckqStack_ = this->eigveckqStack_test[iq];

                uint negativeqindex = -1; // system->findEquivalentPointBZ(-q, Ncells);

                for (uint i = 0; i < Nqpoints; ++i)
                {
                    if (arma::norm(q + q_points.row(i)) < 1E-7)
                    {
                        negativeqindex = i;
                        // std::cout << "For q_index " << iq << "it's negative counterpart is " << negativeqindex << std::endl;
                        break;
                    }
                }

                #pragma omp parallel for
                for (uint ig = 0; ig < nGs * (nGs + 1) / 2; ig++)
                {
                    //uint iq = ik; //indecesqg.row(i_aux)(0);
                    uint g  = indecesqg.row(ig)(1);
                    uint g2 = indecesqg.row(ig)(2);

                    arma::rowvec G = ReciprocalVectors.row(g);
                    arma::rowvec G2 = ReciprocalVectors.row(g2);
                    
                    uint negativeG = fetchReciprocalLatticeVector(-G);
                    uint negativeG2 = fetchReciprocalLatticeVector(-G2);
                    
                    std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(G, G2, q);

                    this->Chimatrix_.slice(iq).row(g)(g2) = Chi;
                    this->Chimatrix_.slice(iq).row(g2)(g) = std::conj(Chi);
                    
                    this->Chimatrix_.slice(negativeqindex).row(negativeG2)(negativeG) = Chi;
                    this->Chimatrix_.slice(negativeqindex).row(negativeG)(negativeG2) = std::conj(Chi);

                    double potentialg  = std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q)); // coulombFT(g, g, q);
                    double potentialg2 = std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q)); // std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q));

                    double potentialnegativeG  = std::sqrt(coulombFT(negativeG, negativeG, system->kpoints.row(negativeqindex))) * std::sqrt(coulombFT(negativeG2, negativeG2, system->kpoints.row(negativeqindex))); // coulombFT(negativeG, negativeG, system->kpoints.row(negativeqindex));
                    double potentialnegativeG2 = std::sqrt(coulombFT(negativeG, negativeG, system->kpoints.row(negativeqindex))) * std::sqrt(coulombFT(negativeG2, negativeG2, system->kpoints.row(negativeqindex))); // coulombFT(negativeG2, negativeG2, system->kpoints.row(negativeqindex));
                    double kroneckerdelta = g == g2 ? 1 : 0;

                    this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg * this->Chimatrix_.slice(iq).row(g)(g2);
                    this->epsilonmatrix_.slice(iq).row(g2)(g) = kroneckerdelta - potentialg2 * this->Chimatrix_.slice(iq).row(g2)(g);

                    this->epsilonmatrix_.slice(negativeqindex).row(negativeG2)(negativeG) = kroneckerdelta - potentialnegativeG2 * this->Chimatrix_.slice(negativeqindex).row(negativeG2)(negativeG);
                    this->epsilonmatrix_.slice(negativeqindex).row(negativeG)(negativeG2) = kroneckerdelta - potentialnegativeG * this->Chimatrix_.slice(negativeqindex).row(negativeG)(negativeG2);
                }

                percentage = ((double)ik + 1)*100./(double)Nqpoints;

                if ((int) percentage % 5 == 0 && (int) percentage_aux != (int) percentage)
                {
                    std::cout << (int) percentage << "%%, " << std::flush;
                    percentage_aux = percentage;
                }
            }
        }
        
        if (odd == 1){

            double percentage = 0;
            double percentage_aux = 0;
            for (uint ik = 0; ik < nq; ik++)
            {
                uint iq = indecesqg.row(ik*nGs*(nGs + 1)/2)(0); 
                arma::rowvec q = q_points.row(iq);

                this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
                this->eigveckqStack_ = this->eigveckqStack_test[iq];

                #pragma omp parallel for
                for (uint ig = 0; ig < nGs * (nGs + 1) / 2; ig++)
                {
                    uint g = indecesqg.row(ig)(1);
                    uint g2 = indecesqg.row(ig)(2);

                    arma::rowvec G = ReciprocalVectors.row(g);
                    arma::rowvec G2 = ReciprocalVectors.row(g2);
                    arma::rowvec q = q_points.row(iq);

                    uint negativeG = fetchReciprocalLatticeVector(-G);
                    uint negativeG2 = fetchReciprocalLatticeVector(-G2);

                    this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
                    this->eigveckqStack_ = this->eigveckqStack_test[iq];

                    std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(G, G2, q);

                    this->Chimatrix_.slice(iq).row(g)(g2) = Chi;
                    this->Chimatrix_.slice(iq).row(g2)(g) = std::conj(Chi);

                    this->Chimatrix_.slice(Nktotal - iq - 1).row(negativeG2)(negativeG) = Chi;
                    this->Chimatrix_.slice(Nktotal - iq - 1).row(negativeG)(negativeG2) = std::conj(Chi);

                    double potentialg = std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g, g, q)); // coulombFT(g, g, system->kpoints.row(iq));
                    double potentialg2 = std::sqrt(coulombFT(g2, g2, q)) * std::sqrt(coulombFT(g2, g2, q)); // coulombFT(g, g, system->kpoints.row(iq));

                    double potentialnegativeG = std::sqrt(coulombFT(negativeG, negativeG, q_points.row(Nktotal - iq - 1))) * std::sqrt(coulombFT(negativeG, negativeG, q_points.row(Nktotal - iq - 1)));
                    double potentialnegativeG2 = std::sqrt(coulombFT(negativeG2, negativeG2, q_points.row(Nktotal - iq - 1))) * std::sqrt(coulombFT(negativeG2, negativeG2, q_points.row(Nktotal - iq - 1)));
                    
                    double kroneckerdelta = g == g2 ? 1 : 0;

                    this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg * Chi;
                    this->epsilonmatrix_.slice(iq).row(g2)(g) = kroneckerdelta - potentialg2 * std::conj(Chi);

                    this->epsilonmatrix_.slice(Nktotal - iq - 1).row(negativeG2)(negativeG) = kroneckerdelta - potentialnegativeG2 * Chi;
                    this->epsilonmatrix_.slice(Nktotal - iq - 1).row(negativeG)(negativeG2) = kroneckerdelta - potentialnegativeG * std::conj(Chi);
                }

                percentage = ((double)ik + 1) * 100 / nq;

                if ((int) percentage % 5 == 0 && (int) percentage_aux != (int) percentage)
                {
                    std::cout << (int) percentage << "%%, " << std::flush;
                    percentage_aux = percentage;
                }
            }
        }
    }


    auto stop_dielectric_matrix_mesh = high_resolution_clock::now();
    auto duration_dielectric_matrix_mesh = duration_cast<milliseconds>(stop_dielectric_matrix_mesh - start);

    std::cout << "Done. \nCalculation of dielectric matrix done in " << duration_dielectric_matrix_mesh.count()/1000.0  << " s." << std::endl << std::flush;
}

/**
 * Method to compute the static Q2D dielectric matrix in the BZ mesh.
 * @return void
*/
void ExcitonTB::compute_quasi2D_DielectricMatrix(){

    auto start = high_resolution_clock::now();

    if (this->mode == "realspace"){
        std::cout << "Numerical screening in real space not available. Terminating" << std::endl;
        exit(0);
    }
    
    if (this->mode == "reciprocalspace"){

        arma::mat ReciprocalVectors = this->trunreciprocalLattice_;
        uint nGs = ReciprocalVectors.n_rows;

        std::cout << "G vectors included in the calculation: " << std::endl;
        printReciprocalLattice();

        this->setq_points_list(system->kpoints); // The set of q points where the dielectric matrix is computed coincides with the BZ mesh
        uint Nqpoints = this->qpoints_list_.n_rows;
        uint nk = Nqpoints;
        uint Nktotal = system->nk;
        
        uint Ncells = this->ncell_;      
        uint odd = Ncells % 2;
        uint nq = odd == 1 ? Ncells * (Ncells - odd) / 2 + (Ncells - odd) / 2 + 1 : (2 * Ncells - 1) + (Ncells - 1) * (Ncells - 2) / 2 + Ncells / 2 + 1; // Only half of the BZ
        
        arma::mat q_points(nq, 3, arma::fill::zeros);
        q_points = this->qpoints_list_; 

        uint ncell_aux = this->ncell_aux;
        uint nk_aux = this->nk_aux;

        double d = this->d_;
        
        uint basisdim = system->basisdim;
        
        std::cout << "Diagonalizing H(k+q) for every point q, for every point k... " << std::flush;

        if (this->eigvalkqStack_test.is_empty() || this->eigveckqStack_test.empty())
        {
            this->eigvalkqStack_test = arma::cube(basisdim, nk_aux, Nqpoints, arma::fill::zeros);

            std::vector<arma::cx_cube> vector_aux;
            vector_aux.resize(Nqpoints);
            this->eigveckqStack_test = vector_aux;

            for (uint iq = 0; iq < Nqpoints; ++iq)
            {
                this->eigveckqStack_test[iq] = arma::cx_cube(basisdim, basisdim, nk_aux, arma::fill::zeros);
            }
        }
        else
        {
            this->eigvalkqStack_test.set_size(basisdim, nk_aux, Nqpoints);
            this->eigveckqStack_test.resize(Nqpoints);

            for (uint iq = 0; iq < Nqpoints; ++iq)
            {
                this->eigveckqStack_test[iq] = arma::cx_cube(basisdim, basisdim, nk_aux, arma::fill::zeros);
            }
        }

        arma::vec auxEigVal(basisdim);
        arma::cx_mat auxEigvec(basisdim, basisdim);
        
        std::cout << "q point " << std::flush;

        for (uint iq = 0; iq < Nqpoints; ++iq)
        {
            arma::rowvec q = q_points.row(iq);

            for (uint i = 0; i < nk_aux; i++)
            {
                arma::rowvec kq = this->kpoints_aux.row(i) + q;
                system->solveBands(kq, auxEigVal, auxEigvec);

                auxEigvec = fixGlobalPhase(auxEigvec);
                this->eigvalkqStack_test.slice(iq).col(i) = auxEigVal;
                this->eigveckqStack_test[iq].slice(i) = auxEigvec;
            }

            std::cout << iq + 1 << ", " << std::flush;
        }

        std::cout << "\nDone. \nComputing Q2D dielectric matrix in the BZ mesh... \n" << std::flush;

        arma::cx_mat auxvecsol(nGs,nGs,arma::fill::zeros);
        arma::cx_mat auxvec(nGs,nGs,arma::fill::eye);

        /*arma::imat indecesqg(nk*nGs*nGs,3,arma::fill::zeros);

        int index = 0;
        for (uint iq = 0; iq < nk; iq++){
            for (uint g = 0; g < nGs; g++){
                for (uint g2 = 0; g2 < nGs; g2++){
                    indecesqg.row(index)(0) = iq;
                    indecesqg.row(index)(1) = g;
                    indecesqg.row(index)(2) = g2;
                    index++;
                }
            }
        }*/

        arma::imat indecesqg(nq*nGs*(nGs+1)/2,3,arma::fill::zeros);

        if (odd == 0){ // If Ncells is even, then q points at the boundaries of the BZ with no symmetric counterpart are computed seperately
            int i=0;
            
            for (uint iq = 0; iq < Ncells; iq++){ // 1st BZ boundary
                for (uint g = 0; g < nGs; g++){
                    for (uint g2 = g; g2 < nGs; g2++){
                        indecesqg.row(i)(0) = iq;
                        indecesqg.row(i)(1) = g;
                        indecesqg.row(i)(2) = g2;
                        i++;
                    }
                }
            }

            for (uint iq = 1; iq < Ncells; iq++){ // 2nd BZ boundary
                for (uint g = 0; g < nGs; g++){
                    for (uint g2 = g; g2 < nGs; g2++){
                        indecesqg.row(i)(0) = iq*Ncells;
                        indecesqg.row(i)(1) = g;
                        indecesqg.row(i)(2) = g2;
                        i++;
                    }
                }
            }

            for (uint iq = 1; iq < Ncells/2; iq++){ // Center-symmetric submesh of the full BZ mesh
                for (uint iq2 = 1; iq2 < Ncells; iq2++){    
                    for (uint g = 0; g < nGs; g++){
                        for (uint g2 = g; g2 < nGs; g2++){
                            indecesqg.row(i)(0) = iq2 + iq*Ncells;
                            indecesqg.row(i)(1) = g;
                            indecesqg.row(i)(2) = g2;
                            i++;
                        }
                    }
                }
            }

            for (uint iq = 0; iq < Ncells/2; iq++){ // Center-symmetric submesh of the full BZ mesh
                for (uint g = 0; g < nGs; g++){
                    for (uint g2 = g; g2 < nGs; g2++){
                        indecesqg.row(i)(0) = 1 + iq + Ncells*Ncells/2;
                        indecesqg.row(i)(1) = g;
                        indecesqg.row(i)(2) = g2;
                        i++;
                    }
                }
            }
        } else if (odd == 1) { // If Ncells is odd, the full BZ mesh is automatically center-symmetric
            int i=0;
            
            for (uint iq = 0; iq < nq; iq++){
                for (uint g = 0; g < nGs; g++){
                    for (uint g2 = g; g2 < nGs; g2++){
                        indecesqg.row(i)(0) = iq;
                        indecesqg.row(i)(1) = g;
                        indecesqg.row(i)(2) = g2;
                        i++;
                    }
                }
            }
        }

        std::cout << "Calculation of dielectric matrix at " << std::endl;

        double percentage = 0;
        double percentage_aux = 0;
        
        /*for (uint ik = 0; ik < nk; ik++)
        {
            uint iq = indecesqg.row(ik*nGs*nGs)(0);
            arma::rowvec q = q_points.row(iq);

            this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
            this->eigveckqStack_ = this->eigveckqStack_test[iq];

            #pragma omp parallel for
            for (uint ig = 0; ig < nGs * nGs; ig++)
            {
                uint g  = indecesqg.row(ig)(1);
                uint g2 = indecesqg.row(ig)(2);

                arma::rowvec G = ReciprocalVectors.row(g);
                arma::rowvec G2 = ReciprocalVectors.row(g2);

                std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(G, G2, q, d);

                double potentialg =  std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q)); // coulombFT(g, g, q);

                this->Chimatrix_.slice(iq).row(g)(g2) = Chi;                    

                double kroneckerdelta = g == g2 ? 1 : 0;

                this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg * Chi;                    
            }

            percentage = ((double)ik + 1)*100./(double)Nqpoints;

            if (ik % 5 == 0)
            {
                std::cout << (int) percentage << "%%, " << std::flush;
                percentage_aux = percentage;
            }
        }*/

        if (odd == 0){

            std::cout << " (Number of k points is even. Computing the dielectric matrix at the nonsymmetric part of the BZ... )" << std::endl;
            uint iq_test = 0;
            arma::rowvec q_test = q_points.row(iq_test);

            double percentage = 0;
            double percentage_aux = 0;
            
            for (uint ik = 0; ik < (2 * Ncells - 1); ik++)
            {
                uint iq = indecesqg.row(ik*nGs*(nGs+1)/2)(0);
                arma::rowvec q = q_points.row(iq);

                this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
                this->eigveckqStack_ = this->eigveckqStack_test[iq];

                #pragma omp parallel for
                for (uint ig = 0; ig < nGs * (nGs + 1) / 2; ig++)
                {
                    uint g  = indecesqg.row(ig)(1);
                    uint g2 = indecesqg.row(ig)(2);

                    arma::rowvec G = ReciprocalVectors.row(g);
                    arma::rowvec G2 = ReciprocalVectors.row(g2);

                    std::complex<double> Chi = compute_quasi2D_PolarizabilityMatrixElement(G, G2, q, d);

                    double potentialg =  std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q)); // coulombFT(g, g, q);

                    this->Chimatrix_.slice(iq).row(g)(g2) = Chi;

                    double kroneckerdelta = g == g2 ? 1 : 0;

                    std::complex<double> epsilon_gg2_element = kroneckerdelta - potentialg * Chi;

                    this->epsilonmatrix_.slice(iq).row(g)(g2) = epsilon_gg2_element;
                    this->epsilonmatrix_.slice(iq).row(g2)(g) = std::conj(epsilon_gg2_element);
                }

                percentage = ((double)ik + 1)*100./(double)Nqpoints;

                if ((int) percentage % 5 == 0 && (int) percentage_aux != (int) percentage)
                {
                    std::cout << (int) percentage << "%%, " << std::flush;
                    percentage_aux = percentage;
                }
            }

            // Computes the dielectric matrix at the centrosymmetric submesh of the BZ

            std::cout << " (Computing the dielectric matrix at the centrosymmetric part of the BZ... )" << std::endl;

            uint iq_i = (2 * Ncells - 1);
            uint ik_f = (Ncells - 1)*(Ncells - 2)/2 + Ncells/2;
            for (uint ik = 0; ik < ik_f; ik++)
            {
                uint iq = indecesqg.row(iq_i*nGs*(nGs + 1)/2 + ik*nGs*(nGs + 1)/2)(0); // indecesqg.row(i_aux)(0);
                arma::rowvec q = q_points.row(iq);

                this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
                this->eigveckqStack_ = this->eigveckqStack_test[iq];

                uint negativeqindex = -1; // system->findEquivalentPointBZ(-q, Ncells);

                for (uint i = 0; i < Nqpoints; ++i)
                {
                    if (arma::norm(q + q_points.row(i)) < 1E-7)
                    {
                        negativeqindex = i;
                        // std::cout << "For q_index " << iq << "it's negative counterpart is " << negativeqindex << std::endl;
                        break;
                    }
                }

                #pragma omp parallel for
                for (uint ig = 0; ig < nGs * (nGs + 1) / 2; ig++)
                {
                    //uint iq = ik; //indecesqg.row(i_aux)(0);
                    uint g  = indecesqg.row(ig)(1);
                    uint g2 = indecesqg.row(ig)(2);

                    arma::rowvec G = ReciprocalVectors.row(g);
                    arma::rowvec G2 = ReciprocalVectors.row(g2);
                    
                    uint negativeG = fetchReciprocalLatticeVector(-G);
                    uint negativeG2 = fetchReciprocalLatticeVector(-G2);
                    
                    std::complex<double> Chi = compute_quasi2D_PolarizabilityMatrixElement(G, G2, q, d);

                    this->Chimatrix_.slice(iq).row(g)(g2) = Chi;
                    this->Chimatrix_.slice(iq).row(g2)(g) = std::conj(Chi);
                    
                    this->Chimatrix_.slice(negativeqindex).row(negativeG2)(negativeG) = Chi;
                    this->Chimatrix_.slice(negativeqindex).row(negativeG)(negativeG2) = std::conj(Chi);

                    double potentialg  = std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q)); // coulombFT(g, g, q);
                    double potentialg2 = std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q)); // std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q));

                    double potentialnegativeG  = std::sqrt(coulombFT(negativeG, negativeG, system->kpoints.row(negativeqindex))) * std::sqrt(coulombFT(negativeG2, negativeG2, system->kpoints.row(negativeqindex))); // coulombFT(negativeG, negativeG, system->kpoints.row(negativeqindex));
                    double potentialnegativeG2 = potentialnegativeG;
                    double kroneckerdelta = g == g2 ? 1 : 0;

                    this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg * this->Chimatrix_.slice(iq).row(g)(g2);
                    this->epsilonmatrix_.slice(iq).row(g2)(g) = kroneckerdelta - potentialg2 * this->Chimatrix_.slice(iq).row(g2)(g);

                    this->epsilonmatrix_.slice(negativeqindex).row(negativeG2)(negativeG) = kroneckerdelta - potentialnegativeG2 * this->Chimatrix_.slice(negativeqindex).row(negativeG2)(negativeG);
                    this->epsilonmatrix_.slice(negativeqindex).row(negativeG)(negativeG2) = kroneckerdelta - potentialnegativeG * this->Chimatrix_.slice(negativeqindex).row(negativeG)(negativeG2);
                }

                percentage = ((double)ik + 1)*100./(double)Nqpoints;

                if ((int) percentage % 5 == 0) {
                    std::cout << (int) percentage << "%" << std::endl;
                }
            }
        }
        
        if (odd == 1){

            double percentage = 0;
            double percentage_aux = 0;
            for (uint ik = 0; ik < nq; ik++)
            {
                uint iq = indecesqg.row(ik*nGs*(nGs + 1)/2)(0); 
                arma::rowvec q = q_points.row(iq);

                this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
                this->eigveckqStack_ = this->eigveckqStack_test[iq];

                #pragma omp parallel for
                for (uint ig = 0; ig < nGs * (nGs + 1) / 2; ig++)
                {
                    uint g = indecesqg.row(ig)(1);
                    uint g2 = indecesqg.row(ig)(2);

                    arma::rowvec G = ReciprocalVectors.row(g);
                    arma::rowvec G2 = ReciprocalVectors.row(g2);
                    arma::rowvec q = q_points.row(iq);

                    uint negativeG = fetchReciprocalLatticeVector(-G);
                    uint negativeG2 = fetchReciprocalLatticeVector(-G2);

                    this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
                    this->eigveckqStack_ = this->eigveckqStack_test[iq];

                    std::complex<double> Chi = compute_quasi2D_PolarizabilityMatrixElement(G, G2, q, d);

                    this->Chimatrix_.slice(iq).row(g)(g2) = Chi;
                    this->Chimatrix_.slice(iq).row(g2)(g) = std::conj(Chi);

                    this->Chimatrix_.slice(Nktotal - iq - 1).row(negativeG2)(negativeG) = Chi;
                    this->Chimatrix_.slice(Nktotal - iq - 1).row(negativeG)(negativeG2) = std::conj(Chi);

                    double potentialg = std::sqrt(coulombFT(g, g, q)) * std::sqrt(coulombFT(g2, g2, q)); // coulombFT(g, g, system->kpoints.row(iq));
                    double potentialg2 = potentialg;

                    double potentialnegativeG = std::sqrt(coulombFT(negativeG, negativeG, q_points.row(Nktotal - iq - 1))) * std::sqrt(coulombFT(negativeG2, negativeG2, q_points.row(Nktotal - iq - 1)));
                    double potentialnegativeG2 = potentialnegativeG;
                    
                    double kroneckerdelta = g == g2 ? 1 : 0;

                    this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg * Chi;
                    this->epsilonmatrix_.slice(iq).row(g2)(g) = kroneckerdelta - potentialg2 * std::conj(Chi);

                    this->epsilonmatrix_.slice(Nktotal - iq - 1).row(negativeG2)(negativeG) = kroneckerdelta - potentialnegativeG2 * Chi;
                    this->epsilonmatrix_.slice(Nktotal - iq - 1).row(negativeG)(negativeG2) = kroneckerdelta - potentialnegativeG * std::conj(Chi);
                }

                percentage = ((double)ik + 1) * 100 / (double)nq;

                if ((int) percentage % 5 == 0) {
                    std::cout << (int) percentage << "%" << std::endl;
                }
            }
        }
    }

    auto stop_dielectric_matrix_mesh = high_resolution_clock::now();
    auto duration_dielectric_matrix_mesh = duration_cast<milliseconds>(stop_dielectric_matrix_mesh - start);

    std::cout << "Done. \nCalculation of dielectric matrix done in " << duration_dielectric_matrix_mesh.count()/1000.0  << " s." << std::endl << std::flush;
}

/**
 * Method to compute the strictly 2D static polarizability matrix and dielectric matrix in a set of user specified q points. More useful for isotropic systems.
 * @return void
*/
void ExcitonTB::compute_2D_DielectricMatrix(std::string kpointsfile){

    auto start = high_resolution_clock::now();

    if (this->mode == "realspace"){

        std::cout << "Implementation of the exciton in real space in progress. Exiting." << std::endl;

        std::exit(0);
    }
    
    if (this->mode == "reciprocalspace"){

        // Reads q points from file
        std::ifstream inputfile(kpointsfile);
        if (!inputfile) {
            std::cout << "File for k points failed to open or does not exist. Exiting" << std::endl;
            exit(1);
        }

        std::string line;
        double qx, qy, qz;
        arma::mat q_points;

        try{
            while(std::getline(inputfile, line)){
                std::istringstream iss(line);
                iss >> qx >> qy >> qz;
                arma::rowvec qpoint{qx, qy, qz};
                q_points.insert_rows(q_points.n_rows,qpoint);
            }
            inputfile.close();
        }
        catch(const std::exception& e){
            std::cerr << e.what() << std::endl;
        }

        setq_points_list(q_points);

        uint Nqpoints = q_points.n_rows;

        std::cout << "Nqpoints = " << Nqpoints << std::endl;

        std::cout << "Computing dielectric matrix in the specified q points... \n" << std::flush;

        uint nk = this->nk_aux;
        uint basisdim = system->basisdim;

        arma::mat ReciprocalVectors = this->trunreciprocalLattice_;
        uint nGs = ReciprocalVectors.n_rows;

        // In case the polarizability/dielectric matrix have been computed before with another routine, reshape to account for a different number of k points
        if (this->Chimatrix_.is_empty()) {
            this->Chimatrix_ = arma::cx_cube(nGs,nGs,Nqpoints,arma::fill::zeros);
        } else {
            this->Chimatrix_.reshape(nGs,nGs,Nqpoints);
        }

        if (this->epsilonmatrix_.is_empty()) {
            this->epsilonmatrix_ = arma::cx_cube(nGs,nGs,Nqpoints,arma::fill::zeros);
        } else {
            this->epsilonmatrix_.reshape(nGs,nGs,Nqpoints);
        }

        arma::cx_cube epsilonmatrix(Nqpoints,nGs,nGs,arma::fill::zeros);
        arma::cx_mat auxvec(nGs,nGs,arma::fill::eye);

        arma::imat indecesqg(Nqpoints*nGs*(nGs+1)/2,3,arma::fill::zeros);

        arma::imat indecesg(nGs*(nGs+1)/2,2,arma::fill::zeros);

        // Generates all the combinations of (k,G,G') indices
        uint i = 0;
        //for (uint iq = 0; iq < Nqpoints; iq++){
            for (uint g = 0; g < nGs; g++){
                for (uint g2 = g; g2 < nGs; g2++){
                    //indecesqg.row(i)(0) = iq;
                    indecesqg.row(i)(0) = g;
                    indecesqg.row(i)(1) = g2;
                    i++;
                }
            }
        //}

        printReciprocalLattice();

        this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
        this->eigvalkqStack_ = arma::mat(basisdim, nk);
        vec auxEigVal(basisdim);
        arma::cx_mat auxEigvec(basisdim, basisdim);

        std::cout << "Computation at\n" << std::flush;

        // These lines are temporary
        // if (this->Invepsilonmatrix_.is_empty()) {
        //     this->Invepsilonmatrix_ = arma::cx_cube(nGs,nGs,Nqpoints,arma::fill::zeros);
        // } else {
        //     this->Invepsilonmatrix_.reshape(nGs,nGs,Nqpoints);
        // }

        for (uint iq = 0; iq < Nqpoints; iq++){  

            arma::rowvec q = q_points.row(iq);

            //std::cout << iq << ", " << std::flush;

            for (uint ik = 0; ik < nk; ik++){
                arma::rowvec kq = this->kpoints_aux.row(ik) + q;
                system->solveBands(kq, auxEigVal, auxEigvec);

                auxEigvec = fixGlobalPhase(auxEigvec);
                this->eigvalkqStack_.col(ik) = auxEigVal;
                this->eigveckqStack_.slice(ik) = auxEigvec;
            };

            #pragma omp parallel for 
            for (uint ig = 0; ig < nGs*(nGs+1)/2; ig++){
                uint g  = indecesqg.row(ig)(0);
                uint g2 = indecesqg.row(ig)(1);

                arma::rowvec G = ReciprocalVectors.row(g);                
                arma::rowvec G2 = ReciprocalVectors.row(g2);

                std::complex<double> Chi = this->compute_2D_PolarizabilityMatrixElement(G, G2, q);

                this->Chimatrix_.slice(iq).row(g)(g2) = Chi;
                this->Chimatrix_.slice(iq).row(g2)(g) = std::conj(Chi);

                double potentialg = std::sqrt(coulomb_2D_FT(q + G)) * std::sqrt(coulomb_2D_FT(q + G2)); // double potentialg = coulomb_2D_FT(q + G);
                double potentialg2 = coulomb_2D_FT(q + G2);
                double kroneckerdelta = g == g2? 1 : 0;

                this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg*Chi;
                this->epsilonmatrix_.slice(iq).row(g2)(g) = kroneckerdelta - potentialg*std::conj(Chi);
            }

            // These lines are temporary
            // std::cout << "Inverting matrix and printing it to file..." << std::endl;

            // std::string aux_file_name = "q_point_" + std::to_string(iq) + "_" + kpointsfile;

            // FILE* textfile = fopen(aux_file_name.c_str(), "w");

            // if (textfile == NULL){
            //     std::cout << "File for inverse of the dielectric matrix failed to open. Exiting" << std::endl;
            //     exit(1);
            // }

            // this->Invepsilonmatrix_.slice(iq) = arma::solve(this->epsilonmatrix_.slice(iq),auxvec);
            // int nGs_cut = 100;
            // //for(uint i = 0; i < Nqp; i++){
            //     for (uint g = 0; g < nGs_cut; g++){
            //         for (uint g2 = 0; g2 < nGs_cut; g2++){
            //             std::complex<double> aux = this->Invepsilonmatrix_.slice(iq).at(g,g2);
            //             fprintf(textfile, "%11.7lf%11.7lf", real(aux), imag(aux));
            //         }
            //         fprintf(textfile, "\n");
            //     }
            // //}

            // fclose(textfile);

            double percentage = ((double)iq + 1.0) / (double)Nqpoints * 100;
            std::cout << percentage << "%" << std::flush;
        }
        // Comment is temporary
        std::cout << "Done.\n" << std::endl;

    }

    auto stop_dielectric_matrix_mesh = high_resolution_clock::now();
    auto duration_dielectric_matrix_mesh = duration_cast<milliseconds>(stop_dielectric_matrix_mesh - start);

    std::cout << "Done in " << duration_dielectric_matrix_mesh.count()/1000.0  << " s." << std::endl << std::flush;
}

/**
 * Method to compute the quasi 2D static polarizability matrix and dielectric matrix in a set of user specified q points. More useful for isotropic systems.
 * @return void
*/
void ExcitonTB::compute_quasi2D_DielectricMatrix(std::string kpointsfile){

    auto start = high_resolution_clock::now();

    if (this->mode == "realspace"){

        std::cout << "Implementation of the exciton in real space in progress. Exiting." << std::endl;

        std::exit(0);
    }
    
    if (this->mode == "reciprocalspace"){

        // Reads q points from file
        std::ifstream inputfile(kpointsfile);
        if (!inputfile) {
            std::cout << "File for k points failed to open or does not exist. Exiting" << std::endl;
            exit(1);
        }

        std::string line;
        double qx, qy, qz;
        arma::mat q_points;
        double d = this->d;

        if(d <= 1E-4){
            d = 0.1;
        }

        try{
            while(std::getline(inputfile, line)){
                std::istringstream iss(line);
                iss >> qx >> qy >> qz;
                arma::rowvec qpoint{qx, qy, qz};
                q_points.insert_rows(q_points.n_rows,qpoint);
            }
            inputfile.close();
        }
        catch(const std::exception& e){
            std::cerr << e.what() << std::endl;
        }

        setq_points_list(q_points);

        uint Nqpoints = q_points.n_rows;

        std::cout << "Nqpoints = " << Nqpoints << std::endl;

        std::cout << "Computing quasi 2D dielectric matrix at the specified q points... \n" << std::flush;

        uint nk = this->nk_aux;
        uint basisdim = system->basisdim;

        arma::mat ReciprocalVectors = this->trunreciprocalLattice_;
        uint nGs = ReciprocalVectors.n_rows;

        // In case the polarizability/dielectric matrix have been computed before with another routine, reshape to account for a different number of k points
        if (this->Chimatrix_.is_empty()) {
            this->Chimatrix_ = arma::cx_cube(nGs,nGs,Nqpoints,arma::fill::zeros);
        } else {
            this->Chimatrix_.reshape(nGs,nGs,Nqpoints);
        }

        if (this->epsilonmatrix_.is_empty()) {
            this->epsilonmatrix_ = arma::cx_cube(nGs,nGs,Nqpoints,arma::fill::zeros);
        } else {
            this->epsilonmatrix_.reshape(nGs,nGs,Nqpoints);
        }

        arma::cx_cube epsilonmatrix(Nqpoints,nGs,nGs,arma::fill::zeros);
        arma::cx_mat auxvec(nGs,nGs,arma::fill::eye);

        arma::imat indecesg(nGs*(nGs+1)/2,2,arma::fill::zeros);

        // Generates all the combinations of (G,G') indices
        uint i = 0;
        for (uint g = 0; g < nGs; g++){
            for (uint g2 = g; g2 < nGs; g2++){
                indecesg.row(i)(0) = g;
                indecesg.row(i)(1) = g2;
                i++;
            }
        }

        printReciprocalLattice();

        this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
        this->eigvalkqStack_ = arma::mat(basisdim, nk);
        vec auxEigVal(basisdim);
        arma::cx_mat auxEigvec(basisdim, basisdim);        

        std::cout << "Computation at\n" << std::flush;

        for (uint iq = 0; iq < Nqpoints; iq++){  

            arma::rowvec q = q_points.row(iq);

            for (uint ik = 0; ik < nk; ik++){
                arma::rowvec kq = this->kpoints_aux.row(ik) + q;
                system->solveBands(kq, auxEigVal, auxEigvec);

                auxEigvec = fixGlobalPhase(auxEigvec);
                this->eigvalkqStack_.col(ik) = auxEigVal;
                this->eigveckqStack_.slice(ik) = auxEigvec;
            };

            #pragma omp parallel for 
            for (uint ig = 0; ig < nGs*(nGs+1)/2; ig++){
                uint g  = indecesg.row(ig)(0);
                uint g2 = indecesg.row(ig)(1);

                arma::rowvec G = ReciprocalVectors.row(g);                
                arma::rowvec G2 = ReciprocalVectors.row(g2);

                std::complex<double> Chi = this->compute_quasi2D_PolarizabilityMatrixElement(G, G2, q, d);                

                double potentialg = std::sqrt(coulomb_2D_FT(q + G))*std::sqrt(coulomb_2D_FT(q + G2)); // double potentialg = coulomb_2D_FT(q + G);

                double kroneckerdelta = g == g2? 1 : 0;

                if (arma::norm(q + G) < 1e-6 || arma::norm(q + G2) < 1e-6){
                    potentialg = 0;
                }

                this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg*Chi;
                this->epsilonmatrix_.slice(iq).row(g2)(g) = kroneckerdelta - potentialg*std::conj(Chi);
            }


            double percentage = ((double)iq + 1.0) / (double)Nqpoints * 100;

            if ((int) percentage % 10 == 0)
            {
                std::cout << (int)percentage << "%" << std::endl;
            }
        }

        std::cout << "Done.\n" << std::endl;
    }

    auto stop_dielectric_matrix_mesh = high_resolution_clock::now();
    auto duration_dielectric_matrix_mesh = duration_cast<milliseconds>(stop_dielectric_matrix_mesh - start);

    std::cout << "Done in " << duration_dielectric_matrix_mesh.count()/1000.0  << " s." << std::endl << std::flush;
}


/**
 * Method to compute the strictly 2D static polarizability matrix in a set of user specified q points. More useful for isotropic systems.
 * @return void
 */
void ExcitonTB::compute_2D_PolarizabilityMatrix(std::string kpointsfile)
{

    auto start = high_resolution_clock::now();

    if (this->mode == "realspace")
    {

        std::cout << "Implementation of the exciton in real space in progress. Exiting." << std::endl;

        std::exit(0);
    }

    if (this->mode == "reciprocalspace")
    {

        // Reads q points from file
        std::ifstream inputfile(kpointsfile);
        if (!inputfile)
        {
            std::cout << "File for k points failed to open or does not exist. Exiting" << std::endl;
            exit(1);
        }

        std::string line;
        double qx, qy, qz;
        arma::mat q_points;

        try
        {
            while (std::getline(inputfile, line))
            {
                std::istringstream iss(line);
                iss >> qx >> qy >> qz;
                arma::rowvec qpoint{qx, qy, qz};
                q_points.insert_rows(q_points.n_rows, qpoint);
            }
            inputfile.close();
        }
        catch (const std::exception &e)
        {
            std::cerr << e.what() << std::endl;
        }

        uint Nqpoints = q_points.n_rows;

        std::cout << "Nqpoints = " << Nqpoints << std::endl;

        std::cout << "Computing polarizability matrix in the specified q points... \n" << std::flush;

        uint nk = this->nk_aux;
        uint basisdim = system->basisdim;

        arma::mat ReciprocalVectors = this->trunreciprocalLattice_;
        uint nGs = ReciprocalVectors.n_rows;

        this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
        this->eigvalkqStack_ = arma::mat(basisdim, nk);
        vec auxEigVal(basisdim);
        arma::cx_mat auxEigvec(basisdim, basisdim);

        // In case the polarizability/dielectric matrix have been computed before with another routine, reshape to account for a different number of k points
        if (this->Chimatrix_.is_empty())
        {
            this->Chimatrix_ = arma::cx_cube(nGs, nGs, Nqpoints, arma::fill::zeros);
        }
        else
        {
            this->Chimatrix_.reshape(nGs, nGs, Nqpoints);
        }

        arma::imat indecesg(nGs*(nGs + 1)/2, 2, arma::fill::zeros);

        // Generates all the combinations of (G,G') indices
        uint i = 0;
        for (uint g = 0; g < nGs; g++)
        {
            for (uint g2 = g; g2 < nGs; g2++)
            {
                indecesg.row(i)(0) = g;
                indecesg.row(i)(1) = g2;
                i++;
            }
        }

        printReciprocalLattice();

        std::cout << "iq = ";

        for (uint iq = 0; iq < Nqpoints; iq++)
        {
            arma::rowvec q = q_points.row(iq);

            std::cout << iq << ", " << std::flush;

            for (uint i = 0; i < nk; i++)
            {
                arma::rowvec kq = this->kpoints_aux.row(i) + q;
                system->solveBands(kq, auxEigVal, auxEigvec);

                auxEigvec = fixGlobalPhase(auxEigvec);
                eigvalkqStack_.col(i) = auxEigVal;
                eigveckqStack_.slice(i) = auxEigvec;
            };

            #pragma omp parallel for
            for (uint i = 0; i < nGs * (nGs + 1) / 2; i++)
            {
                int g = indecesg.row(i)(0);
                int g2 = indecesg.row(i)(1);

                arma::rowvec G = ReciprocalVectors.row(g);
                arma::rowvec G2 = ReciprocalVectors.row(g2);

                std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(G, G2, q);

                this->Chimatrix_.slice(iq).row(g)(g2) = Chi;
                this->Chimatrix_.slice(iq).row(g2)(g) = Chi;
            }
        }

        std::cout << "\n";
    }

    auto stop_dielectric_matrix_mesh = high_resolution_clock::now();
    auto duration_dielectric_matrix_mesh = duration_cast<milliseconds>(stop_dielectric_matrix_mesh - start);

    std::cout << "Done in " << duration_dielectric_matrix_mesh.count() / 1000.0 << " s." << std::endl;
}

/**
 * Method to compute the strictly 2D static polarizability matrix and dielectric matrix at a specific q point from the list of q points. More useful for isotropic systems.
 * @return void
*/
void ExcitonTB::compute_2D_DielectricMatrix_at_q(const arma::rowvec& q){

    uint nk = this->nk_aux;
    int basisdim = system->basisdim;

    arma::mat ReciprocalVectors = this->trunreciprocalLattice_;
    uint nGs = ReciprocalVectors.n_rows;

    this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
    this->eigvalkqStack_ = arma::mat(basisdim, nk);
    vec auxEigVal(basisdim);
    arma::cx_mat auxEigvec(basisdim, basisdim);

    arma::cx_mat Chi0_GG(nGs, nGs, arma::fill::zeros);
    arma::cx_mat epsilon_GG(nGs, nGs, arma::fill::zeros);

    arma::imat indecesg(nGs*(nGs+1)/2,2,arma::fill::zeros);
    uint Nq_points = 1;
    uint iq = 0;

    // In case the polarizability/dielectric matrix have been computed before with another routine, reshape to account for a different number of k points
    if (this->Chimatrix_.is_empty()) {
        this->Chimatrix_ = arma::cx_cube(nGs, nGs, Nq_points, arma::fill::zeros);
    }
    else {
        this->Chimatrix_.reshape(nGs, nGs, Nq_points);
    }

    if (this->epsilonmatrix_.is_empty()) {
        this->epsilonmatrix_ = arma::cx_cube(nGs, nGs, Nq_points, arma::fill::zeros);
    } else {
        this->epsilonmatrix_.reshape(nGs, nGs, Nq_points);
    }

    // Generates all the combinations of (G,G') indices
    uint i = 0;
    
    for (uint g = 0; g < nGs; g++){
        for (uint g2 = g; g2 < nGs; g2++){
            indecesg.row(i)(0) = g;
            indecesg.row(i)(1) = g2;
            i++;
        }
    }

    // In case the kq stack at q is empty, diagonalize H(k + q) for every point k
    if (this->eigvalkqStack_test.is_empty() || this->eigveckqStack_test.size() == 0)
    {
        this->eigvalkqStack_test = arma::cube(basisdim, nk, Nq_points, arma::fill::zeros);
        //this->eigvalkqStack_test.slice(iq) = arma::mat(basisdim, nk, arma::fill::zeros);

        std::vector<arma::cx_cube> vector_aux;
        vector_aux.resize(Nq_points);

        this->eigveckqStack_test.resize(Nq_points);
        this->eigveckqStack_test[iq] = arma::cx_cube(basisdim, basisdim, nk, arma::fill::zeros);

        std::cout << "Diagonalizing H(k+q) for every point k... " << Nq_points << " points.\n" << std::flush;
        for (uint i = 0; i < nk; i++)
        {
            arma::rowvec kq = this->kpoints_aux.row(i) + q;
            system->solveBands(kq, auxEigVal, auxEigvec);
            auxEigvec = fixGlobalPhase(auxEigvec);
            this->eigvalkqStack_test.slice(iq).col(i) = auxEigVal;
            this->eigveckqStack_test[iq].slice(i) = auxEigvec;
        }
        std::cout << "Done.\n" << std::flush;
    }

    this->eigvalkqStack_ = this->eigvalkqStack_test.slice(iq);
    this->eigveckqStack_ = this->eigveckqStack_test[iq];

    #pragma omp parallel for 
    for (uint i = 0; i < nGs*(nGs+1)/2; i++){
        int g  = indecesg.row(i)(0);
        int g2 = indecesg.row(i)(1);

        arma::rowvec G = ReciprocalVectors.row(g);                
        arma::rowvec G2 = ReciprocalVectors.row(g2);

        std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(G, G2, q);

        Chi0_GG.row(g)(g2) = Chi;
        Chi0_GG.row(g2)(g) = std::conj(Chi);

        double potentialg = std::sqrt(coulomb_2D_FT(q + G)) * std::sqrt(coulomb_2D_FT(q + G2)); // double potentialg = coulomb_2D_FT(q + G);
        double potentialg2 = coulomb_2D_FT(q + G2);
        double kroneckerdelta = g == g2? 1 : 0;

        if (arma::norm(q) < 1e-6 && arma::norm(G) < 1e-6){ // At q=0, wing elements (0,G2) are/have to be treated with a regularization at another time, Head element (0,0) is set to 1
            potentialg = 0;
        }

        epsilon_GG.row(g)(g2) = kroneckerdelta - potentialg*Chi;
        epsilon_GG.row(g2)(g) = kroneckerdelta - potentialg*std::conj(Chi);
    }

    this->Chimatrix_.slice(iq) = Chi0_GG;
    this->epsilonmatrix_.slice(iq) = epsilon_GG;
}

/**
 * Method to invert the static dielectric matrix in the list of q points.
 * @return void
*/
void ExcitonTB::invertDielectricMatrix(){

    auto start = high_resolution_clock::now();

    if (this->mode_ == "realspace"){
        std::cout << "Numerical screening in real space not available yet. Terminating." << std::endl;
        std::exit(0);
    }

    if (this->mode_ == "reciprocalspace"){

        uint nGs = this->trunreciprocalLattice_.n_rows;
        uint Nqtotal = this->epsilonmatrix_.n_slices; // The number of q points can be different from the BZ mesh size

        if (Nqtotal < 1) {
            std::cout << "Dielectric matrix must have been computed at least at one point." << std::endl;
            exit(0);
        }

        arma::cx_mat auxvec(nGs,nGs,arma::fill::eye);

        std::cout << "\nInverting the dielectric matrix... " << std::flush;

        if (this->Invepsilonmatrix_.is_empty()) {
            this->Invepsilonmatrix_ = arma::cx_cube(nGs,nGs,Nqtotal,arma::fill::zeros);
        } else {
            this->Invepsilonmatrix_.reshape(nGs,nGs,Nqtotal);
        }

        #pragma omp parallel for
        for (uint iq = 0; iq < Nqtotal; iq++){
            this->Invepsilonmatrix_.slice(iq) = arma::solve(this->epsilonmatrix_.slice(iq),auxvec);
        }
    }

    auto stop = high_resolution_clock::now();
    auto duration_inversion = duration_cast<milliseconds>(stop - start);

    std::cout << "Done in " << duration_inversion.count()/1000.0 << " s." << std::endl;
}

/**
 * Method to compute the (G,G') matrix element of the static dielectric function at the specified momentum vector q.
 * @details Computes the purely 2D version.
 * @return void
*/
void ExcitonTB::computesingleDielectricFunctionMatrixElement() {
    auto start = high_resolution_clock::now();

    if(mode == "realspace"){
        std::cout << "Numerical screening in real space not available yet. Terminating." << std::endl;
        std::exit(0);
    }

    if(mode == "reciprocalspace"){

        printReciprocalLattice();

        std::complex<double> Chi;
        double d = this->d;

        arma::rowvec q = this->q_;

        arma::rowvec g = this->trunreciprocalLattice_.row(this->Gs_(0)); // Sets G
        arma::rowvec g2 = this->trunreciprocalLattice_.row(this->Gs_(1)); // Sets G'

        uint nk = this->nk_aux;
        int basisdim = system->basisdim;
        arma::mat k_points = this->kpoints_aux;     

        this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
        this->eigvalkqStack_ = arma::mat(basisdim, nk);

        vec auxEigVal(basisdim);
        arma::cx_mat auxEigvec(basisdim, basisdim);

        std::cout << "Diagonalizing H0 for all k+q points ... " << std::flush;

        for (uint i = 0; i < nk; i++){
            arma::rowvec kq = k_points.row(i) + q;
            system->solveBands(kq, auxEigVal, auxEigvec);

            auxEigvec = fixGlobalPhase(auxEigvec);
            this->eigvalkqStack_.col(i) = auxEigVal;
            this->eigveckqStack_.slice(i) = auxEigvec;
        };

        Chi = computesinglePolarizabilityMatrixElement(q, g, g2);

        double potential = std::sqrt(coulomb_2D_FT(q + g))*std::sqrt(coulomb_2D_FT(q + g2));

        double kroneckerdelta = this->Gs_(0) == this->Gs_(1) ? 1 : 0;

        std::complex<long double> epsilon =  kroneckerdelta - potential*Chi;

        std::cout << "\nepsilon_2D(q = " << std::setprecision(10) << q(0) << "," << q(1) << "," << q(2) << ") = " << std::setprecision(30) << std::real(epsilon) << " + i" << std::imag(epsilon) << std::endl;

        std::cout << "\nepsilon_Q2D(d = " << d << ") = " << std::setprecision(17) << this->compute_quasi2D_DielectricMatrixElement(g, g2, q, d) << std::endl;
    }
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Done in " << duration.count()/1000.0 << " s." << std::endl;
}

/**
 * Method to compute the (G,G') matrix element of the static dielectric function at the specified momentum vector q.
 * @details It creates a file with the name "[systemName].screening" where the dielectric function matrix elements are stored.
 * @param kpointsfile File with the kpoints where we want to obtain the bands. If empty or not specified, then the set of 
 * kpoints coincides with the kmesh
 * @return void
*/
void ExcitonTB::computesingleInverseDielectricMatrix(std::string label) {

    auto start = high_resolution_clock::now();

    if(mode == "realspace"){
        std::cout << "Real space dielectric function not implemented yet. Exiting." << std::endl;

        std::exit(0);
    }

    arma::mat ReciprocalVectors = this->trunreciprocalLattice_;
    uint nGs = ReciprocalVectors.n_rows;
    std::string to_continue = "_";

    printReciprocalLattice();

    //continueprompt("The number of reciprocal vectors included is: " + std::to_string(nGs) + ". Do you wish to procceed?[y/n]\n");

    std::string filename_dielectric = label + "_invepsilon.dat";
    FILE* textfile_dielectric = fopen(filename_dielectric.c_str(), "w");

    if (textfile_dielectric == NULL){
        std::cout << "File for inverse of the dielectric matrix failed to open. Exiting" << std::endl;
        exit(1);
    }

    arma::cx_mat dielectric_matrix(nGs, nGs, arma::fill::zeros);

    arma::rowvec q_aux = this->q_;

    //int iq = system_->findEquivalentPointBZ(this->q_, this->ncell);//system_->findEquivalentPointBZ(arma::rowvec(3,arma::fill::zeros), this->ncell); // this has to be changed after when I find the time
    std::cout << "Computing inverse of dielectric matrix at momentum: q = " << this->q_ << std::endl;  
    arma::cx_mat auxvecsol(nGs,nGs,arma::fill::zeros);
    arma::cx_mat auxvec(nGs,nGs,arma::fill::eye);
    uint iGmax = nGs*(nGs+1)/2;
    arma::imat indecesg(iGmax,2,arma::fill::zeros);
    
    int i = 0;
    for (uint g = 0; g < nGs; g++){
        for (uint g2 = g; g2 < nGs; g2++){
            indecesg.row(i)(0) = g;
            indecesg.row(i)(1) = g2;
            i++;
        }
    }

    uint basisdim = system->basisdim;
    vec auxEigVal(basisdim);
    arma::cx_mat auxEigvec(basisdim, basisdim);
    int nk = this->nk_aux;

    double d = this->d;

    // If the eigvalkqStack_ and eigveckqStack_ are empty, reshape them
    if (this->eigveckqStack_.is_empty() && this->eigvalkqStack_.is_empty())
    {
        this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
        this->eigvalkqStack_ = arma::mat(basisdim, nk);
    }
    else if ((this->eigveckqStack_.n_slices != nk || this->eigvalkqStack_.n_rows != basisdim || this->eigvalkqStack_.n_cols != basisdim) || (this->eigvalKQStack_.n_rows != basisdim || this->eigvalKQStack_.n_cols != nk))
    {
        this->eigveckqStack_.reshape(basisdim, basisdim, nk);
        this->eigvalkqStack_.reshape(basisdim, nk);
    }

    std::cout << "Diagonalizing H(k+q) for every point k... " << std::flush;
        for (uint i = 0; i < nk; i++)
        {
            arma::rowvec kq = this->kpoints_aux.row(i) + q_aux;
            system->solveBands(kq, auxEigVal, auxEigvec);
            auxEigvec = fixGlobalPhase(auxEigvec);
            this->eigvalkqStack_.col(i) = auxEigVal;
            this->eigveckqStack_.slice(i) = auxEigvec;
        }
        std::cout << "Done.\n" << std::flush;    
        
    #pragma omp parallel for
    for (uint i = 0; i < iGmax; i++){

        int g = indecesg.row(i)(0);
        int g2 = indecesg.row(i)(1);

        arma::rowvec G = ReciprocalVectors.row(g);                
        arma::rowvec G2 = ReciprocalVectors.row(g2);

        // std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(G, G2, q_aux);
        std::complex<double> Chi = compute_quasi2D_PolarizabilityMatrixElement(G, G2, q_aux, d);
        
        double potentialg = std::sqrt(coulombFT(g, g, q_aux))*std::sqrt(coulombFT(g2, g2, q_aux));        
        double kroneckerdelta = g == g2? 1 : 0;
        
        dielectric_matrix.row(g)(g2) = kroneckerdelta - potentialg*Chi;
        dielectric_matrix.row(g2)(g) = kroneckerdelta - potentialg*std::conj(Chi);
    }

    auxvecsol = arma::solve(dielectric_matrix,auxvec);

    for (unsigned int g = 0; g < nGs; g++){
        for (unsigned int g2 = 0; g2 < nGs; g2++){
            fprintf(textfile_dielectric, "%11.7lf%11.7lf", real(auxvecsol.row(g)(g2)), imag(auxvecsol.row(g)(g2)));
        }
        fprintf(textfile_dielectric, "\n");
    }

    fclose(textfile_dielectric);

    // for (unsigned int g = 0; g < nGs; g++){
    //     for (unsigned int g2 = g; g2 < nGs; g2++){
    //         std::cout << "W(" << g << "," << g2 << ") = " << auxvecsol.row(g)(g2)*coulombFT(g2,g2,system->kpoints.row(iq)) << " ?=? " << auxvecsol.row(g2)(g)*coulombFT(g,g,system->kpoints.row(iq)) << std::endl; 
    //     }
    // }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Done in " << duration.count()/1000.0 << " s.\n" << std::flush;
}

/**
 * Method to compute the regularization of the screened potential W_00(q) at q = 0
 * @details Computes numerically the head element of the dielectric matrix at q = 0
 */
void ExcitonTB::compute_ScreenedPotential_regularization(bool is_system_isotropic)
{
    std::cout << "\nComputing regularization for term W00(q = 0)... " << std::flush;

    // Takes the k vector of the BZ for the exciton closest to the origin
    arma::mat k_mat_aux = system->kpoints;
    sortVectors(k_mat_aux);
    arma::rowvec q0 = this->percentage*k_mat_aux.row(1); 


    uint basisdim = this->system->basisdim;
    uint nk = this->nk_aux;

    uint Nqpoints = this->qpoints_list_.n_rows;
    int origin_index = -1;

    double q0_norm = arma::norm(q0);
    arma::rowvec null_vector = {0., 0., 0.};

    std::cout << "\nchosen percentage = " << this->percentage << "." <<std::endl;
    std::cout << "regularization radius q0 = " << q0_norm << "." << std::endl;

    arma::mat ReciprocalVectors = this->trunreciprocalLattice_;
    uint nGs = ReciprocalVectors.n_rows;

    if (this->percentage == 0.0) {
        this->W00_at_0_ = 0.0;
        std::cout << "W00(q = 0) = " << this->W00_at_0_ << " eV." << std::endl;
        return;
    }

    if (this->potential_ == "keldysh"){

        this->W00_at_0_ = (2 + this->r0*q0_norm) * ec * 1E10 / (2 * eps0 * q0_norm * system->unitCellArea);

        std::cout << "W00(q = 0) = " << this->W00_at_0_ << " eV." << std::endl;

        std::cout << "Done.\n" << std::endl;
        return;
    }

    // Calculates the head element of the dielectric matrix at q = 0 infinitesimal

    vec auxEigVal(basisdim);
    arma::cx_mat auxEigvec(basisdim, basisdim);

    // If the eigvalkqStack_ and eigveckqStack_ are empty, reshape them
    if (this->eigveckqStack_.is_empty() && this->eigvalkqStack_.is_empty())
    {
        this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
        this->eigvalkqStack_ = arma::mat(basisdim, nk);
    }
    else if ((this->eigveckqStack_.n_slices != nk || this->eigvalkqStack_.n_rows != basisdim || this->eigvalkqStack_.n_cols != basisdim) || (this->eigvalKQStack_.n_rows != basisdim || this->eigvalKQStack_.n_cols != nk))
    {
        this->eigveckqStack_.reshape(basisdim, basisdim, nk);
        this->eigvalkqStack_.reshape(basisdim, nk);
    }

    std::cout << "Diagonalizing H(k+q0) for every point k... " << std::flush;

    for (uint i = 0; i < nk; i++)
    {
        arma::rowvec kq = this->kpoints_aux.row(i) + q0;
        system->solveBands(kq, auxEigVal, auxEigvec);
        auxEigvec = fixGlobalPhase(auxEigvec);
        this->eigvalkqStack_.col(i) = auxEigVal;
        this->eigveckqStack_.slice(i) = auxEigvec;
    }

    std::cout << "done, " << std::flush;

    std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(null_vector, null_vector, q0);

    double potentialg = coulomb_2D_FT(null_vector + q0);

    // Calculates the inverse dielectric matrix at q0 infinitesimal, to regularize W_00(0)
    // The screening parameter can be estimated from the head element of the direct dielectric matrix
    // arma::cx_mat dielectric_matrix(nGs, nGs, arma::fill::zeros);
    // arma::cx_mat auxvecsol(nGs, nGs, arma::fill::zeros);
    // arma::cx_mat auxvec(nGs, nGs, arma::fill::eye);

    // arma::imat indecesg(nGs*(nGs + 1)/2, 2, arma::fill::zeros);

    // int i = 0;
    // for (uint g = 0; g < nGs; g++)
    // {
    //     for (uint g2 = g; g2 < nGs; g2++)
    //     {
    //         indecesg.row(i)(0) = g;
    //         indecesg.row(i)(1) = g2;
    //         i++;
    //     }
    // }

    
    // #pragma omp parallel for
    // for (uint i = 0; i < nGs*(nGs + 1)/2; i++)
    // {
    //     int g = indecesg.row(i)(0);
    //     int g2 = indecesg.row(i)(1);

    //     arma::rowvec G = ReciprocalVectors.row(g);
    //     arma::rowvec G2 = ReciprocalVectors.row(g2);

    //     std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(G, G2, q0);

    //     double potentialg = std::sqrt(coulombFT(g, g, q0)) * std::sqrt(coulombFT(g2, g2, q0)); // coulombFT(g, g, q0);
    //     // double potentialg2 = coulombFT(g2, g2, q0);
    //     double kroneckerdelta = g == g2 ? 1 : 0;

    //     dielectric_matrix.row(g)(g2) = kroneckerdelta - potentialg*Chi;
    //     dielectric_matrix.row(g2)(g) = kroneckerdelta - potentialg*std::conj(Chi);
    // }

    // auxvecsol = arma::solve(dielectric_matrix, auxvec); // Inverts the dielectric matrix

    std::complex<double> head_element = 1.0 - potentialg*Chi; // auxvecsol(0, 0);

    double Re_head_element = std::real(head_element);
    double slope = (Re_head_element - 1.)/q0_norm;
    this->slope_ = slope;

    double Re_head_element_perp = Re_head_element;

    if (!is_system_isotropic) { // If system is anisotropic, repeat the calculation for a vector perpendicular to q0 with same norm

        arma::rowvec q0_perpendicular = {-q0(1), q0(0), 0};

        std::cout << "diagonalizing H(k+q0_perpendicular) for every point k... " << std::flush;
        for (uint i = 0; i < nk; i++)
        {
            arma::rowvec kq = this->kpoints_aux.row(i) + q0_perpendicular;
            system->solveBands(kq, auxEigVal, auxEigvec);
            auxEigvec = fixGlobalPhase(auxEigvec);
            this->eigvalkqStack_.col(i) = auxEigVal;
            this->eigveckqStack_.slice(i) = auxEigvec;
        }
        std::cout << "done, finishing computing the regularization... " << std::flush;

        std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(null_vector, null_vector, q0_perpendicular);

        double potentialg = coulomb_2D_FT(null_vector + q0_perpendicular); // coulombFT(g, g, q0);

        // The screening parameter can be estimated from the head element of the direct dielectric matrix
        // #pragma omp parallel for
        // for (uint i = 0; i < nGs * (nGs + 1) / 2; i++)
        // {
        //     int g = indecesg.row(i)(0);
        //     int g2 = indecesg.row(i)(1);

        //     arma::rowvec G = ReciprocalVectors.row(g);
        //     arma::rowvec G2 = ReciprocalVectors.row(g2);

        //     std::complex<double> Chi = compute_2D_PolarizabilityMatrixElement(G, G2, q0_perpendicular);

        //     double potentialg = std::sqrt(coulombFT(g, g, q0_perpendicular)) * std::sqrt(coulombFT(g2, g2, q0_perpendicular)); // coulombFT(g, g, q0);
        //     // double potentialg2 = coulombFT(g2, g2, q0);
        //     double kroneckerdelta = g == g2 ? 1 : 0;

        //     dielectric_matrix.row(g)(g2) = kroneckerdelta - potentialg * Chi;
        //     dielectric_matrix.row(g2)(g) = kroneckerdelta - potentialg * std::conj(Chi);
        // }

        // auxvecsol = arma::solve(dielectric_matrix, auxvec); // Inverts the dielectric matrix

        std::complex<double> head_element = 1.0 - potentialg * Chi; // auxvecsol(0, 0);

        Re_head_element_perp = std::real(head_element);
        this->slope_perp_ = (Re_head_element_perp - 1.) / q0_norm;
    } else {
        this->slope_perp_ = this->slope_;
    }

    this->W00_at_0_ = (2 + 0.5*(Re_head_element + Re_head_element_perp - 2)) * ec * 1E10 / (2 * eps0 * q0_norm * system->unitCellArea);
    std::cout << "W00(q = 0) = " << this->W00_at_0_ << " eV." << std::endl;
    std::cout << "regularization computed with success." << std::endl;
}

/**
 * Method to initialize the BSE.
 * @details Calls the more general routine which allows
 * to specify a subset of the complete basis.
 */
void ExcitonTB::BShamiltonian()
{
    arma::imat basis = {};
    BShamiltonian(basis);
}

/**
 * Initialize BSE hamiltonian matrix and kinetic matrix.
 * @details Instead of calculating the energies and coeficients dinamically, which
 * is too expensive, instead we first calculate those for each k, save them
 * in the heap, and then call them consecutively as we build the matrix.
 * Analogously, we calculate the Fourier transform of the potential beforehand,
 * saving it in the stack so that it can be later called in the matrix element
 * calculation.
 * Also note that this routine involves a omp parallelization when building the matrix.
 * @param basis Subset of the exciton basis to build the BSE. If none, defaults to
 * the complete or original basis.
 * @return void
 */
void ExcitonTB::BShamiltonian(const arma::imat& basis){

    arma::imat basisStates = this->basisStates;
    if (!basis.is_empty()){
        basisStates = basis;
    };

    if (((this->potential_ == "rpa" && this->Invepsilonmatrix_.is_zero()) || (this->exchangePotential_ == "rpa" && this->Invepsilonmatrix_.is_zero())) && this->mode == "reciprocalspace"){
        std::cout << "The Bethe-Salpeter Hamiltonian can not be initialized with the 'rpa' potential if the inverse of the dielectric matrix has not been computed. Terminating." << std::endl;
        exit(0);
    }

    uint64_t basisDimBSE = basisStates.n_rows;
    std::cout << "BSE dimension: " << basisDimBSE << std::endl;

    if (this->mode == "reciprocalspace"){

        uint nGs = system->truncateReciprocalSupercell(this->Gc_exciton_).n_rows;
        
        if (this->nReciprocalVectors != nGs) {
            std::cout << "The number of G vectors to compute the exciton " << this->nReciprocalVectors << " does not match with the number of G vectors in the reciprocal sublattice " << nGs  << " for the cutoff " << this->Gc_exciton_  << " angstrom^{-1}." << std::endl; 
        }

        // if (this->Invepsilonmatrix_.is_empty()) {
            this->compute_ScreenedPotential_regularization(this->isotropic);
        // }
    }
    std::cout << "this->nReciprocalVectors: " << this->nReciprocalVectors << std::endl;
    std::cout << "Initializing Bethe-Salpeter matrix... " << std::flush;

    HBS_ = arma::zeros<cx_mat>(basisDimBSE, basisDimBSE);
    
    // To be able to parallelize over the triangular matrix, we build
    uint64_t loopLength = basisDimBSE*(basisDimBSE + 1)/2.;

    // https://stackoverflow.com/questions/242711/algorithm-for-index-numbers-of-triangular-matrix-coefficients
    #pragma omp parallel for
    for(uint64_t n = 0; n < loopLength; n++){

        arma::cx_vec coefsK, coefsK2, coefsKQ, coefsK2Q;

        uint64_t ii = loopLength - 1 - n;
        uint64_t m  = floor((sqrt(8*ii + 1) - 1)/2);
        uint64_t i = basisDimBSE - 1 - m;
        uint64_t j = basisDimBSE - 1 - ii + m*(m+1)/2;
    
        uint32_t k_index = basisStates(i, 2);
        int v = bandToIndex[basisStates(i, 0)];
        int c = bandToIndex[basisStates(i, 1)];
        uint32_t kQ_index = k_index;

        uint32_t k2_index = basisStates(j, 2);
        int v2 = bandToIndex[basisStates(j, 0)];
        int c2 = bandToIndex[basisStates(j, 1)];
        uint32_t k2Q_index = k2_index;

        // Using the atomic gauge
        if(gauge == "atomic"){
            coefsK = system_->latticeToAtomicGauge(eigvecKStack.slice(k_index).col(v), system->kpoints.row(k_index));
            coefsKQ = system_->latticeToAtomicGauge(eigvecKQStack.slice(kQ_index).col(c), system->kpoints.row(kQ_index));
            coefsK2 = system_->latticeToAtomicGauge(eigvecKStack.slice(k2_index).col(v2), system->kpoints.row(k2_index));
            coefsK2Q = system_->latticeToAtomicGauge(eigvecKQStack.slice(k2Q_index).col(c2), system->kpoints.row(k2Q_index));
        }
        else{
            coefsK = eigvecKStack.slice(k_index).col(v);
            coefsKQ = eigvecKQStack.slice(kQ_index).col(c);
            coefsK2 = eigvecKStack.slice(k2_index).col(v2);
            coefsK2Q = eigvecKQStack.slice(k2Q_index).col(c2);
        }

        std::complex<double> D, X = 0.0;
        if (mode == "realspace"){
            uint32_t effective_k_index = system_->findEquivalentPointBZ(system->kpoints.row(k2_index) - system->kpoints.row(k_index), ncell);
            arma::cx_mat motifFT = ftMotifStack.slice(effective_k_index);
            D = realSpaceInteractionTerm(coefsKQ, coefsK2, coefsK2Q, coefsK, motifFT);
            if(this->exchange){
                X = realSpaceInteractionTerm(coefsKQ, coefsK2, coefsK, coefsK2Q, this->ftMotifQ);
            }            
        }
        else if (mode == "reciprocalspace"){
            recpotptr directPotential = selectReciprocalPotential(this->potential_);
            arma::rowvec k = system->kpoints.row(k_index);
            arma::rowvec k2 = system->kpoints.row(k2_index);
            D = reciprocalInteractionTerm(coefsK, coefsK2, coefsKQ, coefsK2Q, k, k2, k + Q, k2 + Q, this->potential_, this->nReciprocalVectors);
            if(this->exchange){
                recpotptr exchangePotential = selectReciprocalPotential(this->exchangePotential_);
                X = reciprocalInteractionTerm(coefsK2Q, coefsK2, coefsKQ, coefsK, k2 + Q, k2, k + Q, k, this->exchangePotential_, this->nReciprocalVectors);
            }
        }
        
        if (i == j){
            HBS_(i, j) = (this->scissor + 
                          eigvalKQStack.col(kQ_index)(c) - eigvalKStack.col(k_index)(v))/2. 
                          - (D - X)/2.;            
        }
        else{
            HBS_(i, j) =  - (D - X);
        };
    }
       
    HBS_ = HBS + HBS.t();
    std::cout << "Done" << std::endl;
};

/**
 * Routine to diagonalize the BSE and return a Result object.
 * @param method Method to diagonalize the BSE, either 'diag' (standard diagonalization) 
 * 'davidson' (iterative diagonalization) or 'sparse' (Lanczos).
 * @param nstates Number of states to be stored from the diagonalization.
 * @return Result object storing the exciton energies and states.
 */ 
ResultTB* ExcitonTB::diagonalizeRaw(std::string method, int nstates){

    if (HBS.empty() || HBS.is_zero()){
        throw std::invalid_argument("diagonalizeRaw(): BSE Hamiltonian is not initialized.");
    }

    std::cout << "Solving BSE with ";
    arma::vec eigval;
    arma::cx_mat eigvec;

    if (method == "diag"){
        std::cout << "exact diagonalization... " << std::flush;
        arma::eig_sym(eigval, eigvec, HBS);
    }
    else if (method == "davidson"){
        std::cout << "Davidson method... " << std::flush;
        davidson_method(eigval, eigvec, HBS, nstates);
    }
    else if (method == "sparse"){
        std::cout << "Lanczos method..." << std::flush;

        arma::cx_vec cx_eigval;
        //arma::eigs_gen(cx_eigval, eigvec, arma::sp_cx_mat(HBS), nstates, "sr");
        eigval = arma::real(cx_eigval);
    }
    
    std::cout << "Done" << std::endl;

    return new ResultTB(this, eigval, eigvec);
}

/**
 * Routine to diagonalize the BSE and return a Result object.
 * @details Wrapper for the diagonalizeRaw method, which returns a raw pointer.
 * We wrap it into a unique pointer to avoid memory leaks.
 * @param method Method to diagonalize the BSE, either 'diag' (standard diagonalization) 
 * 'davidson' (iterative diagonalization) or 'sparse' (Lanczos).
 * @param nstates Number of states to be stored from the diagonalization.
 * @return Result object storing the exciton energies and states.
 */ 
std::unique_ptr<ResultTB> ExcitonTB::diagonalize(std::string method, int nstates){
    return std::unique_ptr<ResultTB>(diagonalizeRaw(method, nstates));
}

// ------------- Routines to compute Fermi Golden Rule -------------

/**
 * Method to compute density of states associated to non-interacting electron-hole pairs.
 * Considers only the bands defined as the basis for excitons.
 * @param energy Energy at which we evaluate the pair DoS.
 * @param delta Broadening used to smooth the DoS.
 * @return DoS at E.
 */
double ExcitonTB::pairDensityOfStates(double energy, double delta) const {
    
    double dos = 0;
    for(int v = 0; v < (int)valenceBands.n_elem; v++){
        for(int c = 0; c < (int)conductionBands.n_elem; c++){
            for(uint i = 0; i < system->nk; i++){

                uint vband = bandToIndex.at(valenceBands(v)); // Unsigned integer 
                uint cband = bandToIndex.at(conductionBands(c));

                double stateEnergy = eigvalKStack.col(i)(cband) - eigvalKStack.col(i)(vband);
                dos += -PI*imag(rGreenF(energy, delta, stateEnergy));
            };
        }
    }
    dos /= (system->a * system->nk);

    return dos;
}


/** 
 * Routine to compute and write to a file the density of states of non-interacting e-h pairs.
 * @param file Pointer to file.
 * @param delta DoS broadening.
 * @param n Number of points on which we evaluate the DoS.
 * @return void
 */
void ExcitonTB::writePairDOS(FILE* file, double delta, int n){

    double eMin = eigvalKStack.min();
    double eMax = eigvalKQStack.max();
    arma::vec energies = arma::linspace(0, (eMax - eMin)*1.1, n);
    for (double energy : energies){
        double dos = pairDensityOfStates(energy, delta);
        fprintf(file, "%f\t%f\n", energy, dos);
    }
}


/**
 * Routine to compute the non-interacting electron-hole edge pair associated to a given energy.
 * @details We run a search algorithm to find which k value matches the given energy.
 * @param energy Energy at which we want the non-interacting e-h pair.
 * @param gapEnergy Values of the gap at all kpoints.
 * @param side Whether to obtain the pair at +k or -k.
 * @return Vec of coefficients in e-h pair basis associated to desired pair.
 */
cx_vec ExcitonTB::ehPairCoefs(double energy, const vec& gapEnergy, std::string side){

    cx_vec coefs = arma::zeros<cx_vec>(system->nk);
    int closestKindex = -1;
    double eDiff;
    double currentEnergy = gapEnergy(0) - energy;

    for(uint n = 1; n < system->nk/2; n++){
        
        eDiff = gapEnergy(n) - energy;
        if(abs(eDiff) < abs(currentEnergy)){
            closestKindex = n;
            currentEnergy = eDiff;
        };
    };
    std::cout << closestKindex << std::endl;
    std::cout << "Selected k: " << system->kpoints(closestKindex) << "\t" << closestKindex << std::endl;
    std::cout << "Closest gap energy: " << gapEnergy(closestKindex) << std::endl;
    // By virtue of band symmetry, we expect n < nk/2
    double dispersion = PI/(16*system->a);
    if(side == "left"){
        coefs(closestKindex) = 1.;
    }
    else if(side == "right"){
        coefs(system->nk - 1 - closestKindex) = 1.;
    }

    std::cout << "Energy gap (-k): " << gapEnergy(closestKindex) << std::endl;
    std::cout << "Energy gap (k): " << gapEnergy(system->nk - 1 - closestKindex) << std::endl;

    return coefs;
};

/**
 * Method to compute the transition rate from one exciton to a general non-interacting electron-hole pair.
 * @param targetExciton Exciton object representing the final system->
 * @param initialState Exciton state (coefficients) from which the transition happens.
 * @param finalState Final exciton state in the transition.
 * @param energy Energy of the initial exciton state.
 * @return Transition rate from initialState to finalState.
 */ 
double ExcitonTB::fermiGoldenRule(const ExcitonTB& targetExciton, 
                                  const arma::cx_vec& initialState, 
                                  const arma::cx_vec& finalState, double energy){

    double transitionRate = 0;
    arma::imat initialBasis = basisStates;
    arma::imat finalBasis = targetExciton.basisStates;
    cx_mat W = arma::zeros<cx_mat>(finalBasis.n_rows, initialBasis.n_rows);

    // -------- Main loop (W initialization) --------
    #pragma omp parallel for schedule(static, 1) collapse(2)
    for (arma::uword i = 0; i < finalBasis.n_rows; i++){
        for (arma::uword j = 0; j < initialBasis.n_rows; j++){

            arma::cx_vec coefsK, coefsK2, coefsKQ, coefsK2Q;

            int vf = targetExciton.bandToIndex.at(finalBasis(i, 0));
            int cf = targetExciton.bandToIndex.at(finalBasis(i, 1));
            double kf_index = finalBasis(i, 2);
            
            int vi = bandToIndex[initialBasis(j, 0)];
            int ci = bandToIndex[initialBasis(j, 1)];
            double ki_index = initialBasis(j, 2);

            // Using the atomic gauge
            if(gauge == "atomic"){
                coefsK = system_->latticeToAtomicGauge(
                    targetExciton.eigvecKStack.slice(kf_index).col(vf), system->kpoints.row(kf_index));
                coefsKQ = system_->latticeToAtomicGauge(
                    targetExciton.eigvecKQStack.slice(kf_index).col(cf), system->kpoints.row(kf_index));
                coefsK2 = system_->latticeToAtomicGauge(
                    eigvecKStack.slice(ki_index).col(vi), system->kpoints.row(ki_index));
                coefsK2Q = system_->latticeToAtomicGauge(
                    eigvecKQStack.slice(ki_index).col(ci), system->kpoints.row(ki_index));
            }
            else{
                coefsK = targetExciton.eigvecKStack.slice(kf_index).col(vf);
                coefsKQ = targetExciton.eigvecKQStack.slice(kf_index).col(cf);
                coefsK2 = eigvecKStack.slice(ki_index).col(vi);
                coefsK2Q = eigvecKQStack.slice(ki_index).col(ci);
            }

            std::complex<double> D, X;
            if (mode == "realspace"){
                int effective_k_index = system_->findEquivalentPointBZ(
                    system->kpoints.row(ki_index) - system->kpoints.row(kf_index), ncell);
                arma::cx_mat motifFT = ftMotifStack.slice(effective_k_index);
                D = realSpaceInteractionTerm(coefsKQ, coefsK2, coefsK2Q, coefsK, motifFT);
                X = 0;

            }
            else if (mode == "reciprocalspace"){
                arma::rowvec k = system->kpoints.row(kf_index);
                arma::rowvec k2 = system->kpoints.row(ki_index);
                recpotptr potential = selectReciprocalPotential(this->potential_); // Not sure at this point the effect of the chosen potential on this calculation
                D = reciprocalInteractionTerm(coefsK, coefsK2, coefsKQ, coefsK2Q, k, k2, k, k2, this->potential_, this->nReciprocalVectors);
                X = 0;
            }
            
            W(i, j) = - (D - X);                
        };
    };

    double delta = 2.4/(2*ncell); // Adjust delta depending on number of k points
    double rho = targetExciton.pairDensityOfStates(energy, delta);
    std::cout << "DoS value: " << rho << endl;
    double hbar = 6.582119624E-16; // Units are eV*s

    transitionRate = 2*PI*std::norm(arma::cdot(finalState, W*initialState))*rho/hbar;

    return transitionRate;
}


/**
 * Method to identify a k point corresponding to a non-interacting electron-hole pair in the defined system
 * with the energy specified.
 * @param targetExciton Exciton object representing the general final states in the transition.
 * @param energy Energy of the initial state.
 * @param side Whether the transition takes place to an state with +k or -k.
 * @param increasing Used to specify whether the gap increases or decreases with k.
 * @return k vector of the equivalent electron-hole pair.
*/
arma::rowvec ExcitonTB::findElectronHolePair(const ExcitonTB& targetExciton, double energy, std::string side, bool increasing){

// First identify k edge e-h pair with same energy as exciton
    double n = 10; // Submeshing
    arma::rowvec min_k, max_k, kmin, kmax;
    int nk = system->nk;
    if (side == "right"){
        min_k = system->kpoints.row(nk/2);
        max_k = - system->kpoints.row(0);
    }
    else if(side == "left"){
        max_k = system->kpoints.row(0);
        min_k = system->kpoints.row(nk/2 - 1);
    }
    
    arma::rowvec k;
    arma::cx_vec coefsK, coefsKQ, auxCoefsK, auxCoefsKQ;
    double threshold = 1E-8;
    arma::vec eigval;
    arma::cx_mat eigvec;
    int currentIndex;
    double currentEnergy = 0, vEnergy, cEnergy, gap, prevGap;
    double prevEnergy = currentEnergy;
    prevGap = 0;
    
    while(abs(currentEnergy - energy) > threshold){
        for(double i = 0; i <= n; i++){
            
            k = min_k * (1 - i/n) + max_k * i/n;
            targetExciton.system->solveBands(k, eigval, eigvec);

            eigval = eigval(targetExciton.bandList);
            vEnergy = eigval(0);

            if(arma::norm(Q) != 0){
                arma::rowvec kQ = k + Q;
                targetExciton.system->solveBands(kQ, eigval, eigvec);

                eigval = eigval(targetExciton.bandList);
            }
            cEnergy = eigval(1);

            gap = cEnergy - vEnergy;
            if (!increasing && (gap <= energy) && (prevGap > energy)){
                currentIndex = i;
                currentEnergy = gap;

                kmin = min_k * (1 - (currentIndex - 1)/n) + max_k * (currentIndex - 1)/n;
                kmax = min_k * (1 - (currentIndex + 1)/n) + max_k * (currentIndex + 1)/n;
            }
            if (increasing && (gap > energy) && (prevGap <= energy)){
                currentIndex = i;
                currentEnergy = gap;

                kmin = min_k * (1 - (currentIndex - 1)/n) + max_k * (currentIndex - 1)/n;
                kmax = min_k * (1 - (currentIndex + 1)/n) + max_k * (currentIndex + 1)/n;
            }
            prevGap = gap;

        }
        k = min_k * (1 - currentIndex/n) + max_k * currentIndex/n;
        min_k = kmin;
        max_k = kmax;
        arma::cout << "Current edge pair energy: " << currentEnergy << arma::endl;
        arma::cout << "Target energy: " << energy << "\n" << arma::endl;

        if (currentEnergy == prevEnergy){
            n += 1;
        }
        prevEnergy = currentEnergy;
    }

    arma::cout << "k: " << k << arma::endl;

    return k;
};

/**
 * Method to compute the transition to an edge e-h pair with the same energy (up to some error) as the bulk exciton.
 * @param targetExciton Exciton object representing the general final states in the transition.
 * @param initialState Exciton state from which the transition happens.
 * @param energy Energy of the initial state.
 * @param side Whether the transition takes place to an state with +k or -k.
 * @param increasing Used to specify whether the gap increases or decreases with k.
 * @return Transition rate from the initial exciton state to a non-interacting e-h pair.
 */
double ExcitonTB::edgeFermiGoldenRule(const ExcitonTB& targetExciton, 
                                      const arma::cx_vec& initialState, 
                                      double energy, std::string side, bool increasing){

    double transitionRate = 0;
    arma::imat initialBasis = basisStates;

    arma::rowvec k = findElectronHolePair(targetExciton, energy, side, increasing);

    arma::vec eigval;
    arma::cx_mat eigvec;
    arma::cx_vec coefsK, coefsKQ;

    targetExciton.system->solveBands(k, eigval, eigvec);

    eigvec = fixGlobalPhase(eigvec);
    eigvec = eigvec.cols(targetExciton.bandList);
    coefsK = eigvec.col(0);

    if(arma::norm(Q) != 0){
        arma::rowvec kQ = k + Q;
        targetExciton.system->solveBands(kQ, eigval, eigvec);

        eigvec = fixGlobalPhase(eigvec);
        eigvec = eigvec.cols(targetExciton.bandList);
    }
    coefsKQ = eigvec.col(1);

    bool computeOccupations = true;
    if (computeOccupations){
        //////// Specific for Bi ribbon; must be deleted afterwards.
        int N = targetExciton.system->basisdim;
        double l_e_edge_occ = arma::norm(coefsKQ.subvec(0, 15));
        double r_e_edge_occ = arma::norm(coefsKQ.subvec(N - 16, N - 1));
        double l_h_edge_occ = arma::norm(coefsK.subvec(0, 15));
        double r_h_edge_occ = arma::norm(coefsK.subvec(N - 16, N - 1));

        std::cout << "left e occ.: " << l_e_edge_occ << "\nright e occ: " << r_e_edge_occ << std::endl;
        std::cout << "Total e occ.: " << std::sqrt(l_e_edge_occ*l_e_edge_occ + r_e_edge_occ*r_e_edge_occ) << arma::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "left h occ.: " << l_h_edge_occ << "\nright h occ: " << r_h_edge_occ << std::endl;
        std::cout << "Total h occ.: " << std::sqrt(l_h_edge_occ*l_h_edge_occ + r_h_edge_occ*r_h_edge_occ) << arma::endl;
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "Total e-h pair edge occu.: " << std::sqrt(l_e_edge_occ*l_e_edge_occ + r_e_edge_occ*r_e_edge_occ) + 
                    std::sqrt(l_h_edge_occ*l_h_edge_occ + r_h_edge_occ*r_h_edge_occ) << std::endl;
    }
    
    // Now compute motif FT using k of edge pair
    double radius = arma::norm(system->bravaisLattice.row(0)) * cutoff_;
    arma::mat cells = system_->truncateSupercell(ncell, radius);
    potptr potential = selectPotential(this->potential_);

    int natoms = system->natoms;
    arma::cx_cube ftMotifStack = arma::cx_cube(natoms, natoms, system->kpoints.n_rows);
    
    #pragma omp parallel for collapse(2)
    for(uint i = 0; i < system->nk; i++){
        for(int fAtomIndex = 0; fAtomIndex < natoms; fAtomIndex++){
            for(int sAtomIndex = fAtomIndex; sAtomIndex < natoms; sAtomIndex++){
                ftMotifStack(fAtomIndex, sAtomIndex, i) = 
                motifFourierTransform(fAtomIndex, sAtomIndex, system->kpoints.row(i) - k, cells, potential);
                ftMotifStack(sAtomIndex, fAtomIndex, i) = conj(ftMotifStack(fAtomIndex, sAtomIndex, i));
            }   
        }
    }
    
    arma::cx_vec W = arma::zeros<arma::cx_vec>(initialBasis.n_rows);

    // -------- Main loop (W initialization) --------
    // #pragma omp parallel for
    for (uint i = 0; i < initialBasis.n_rows; i++){

        arma::cx_vec coefsK2, coefsK2Q;
        
        int vi = bandToIndex[initialBasis(i, 0)];
        int ci = bandToIndex[initialBasis(i, 1)];
        double ki_index = initialBasis(i, 2);

        // Using the atomic gauge
        if(gauge == "atomic"){
            coefsK2 = system_->latticeToAtomicGauge(eigvecKStack.slice(ki_index).col(vi), system->kpoints.row(ki_index));
            coefsK2Q = system_->latticeToAtomicGauge(eigvecKQStack.slice(ki_index).col(ci), system->kpoints.row(ki_index));
        }
        else{
            coefsK2 = eigvecKStack.slice(ki_index).col(vi);
            coefsK2Q = eigvecKQStack.slice(ki_index).col(ci);
        }

        std::complex<double> D, X;
        if (mode == "realspace"){
            arma::cx_mat motifFT = ftMotifStack.slice(ki_index);
            D = realSpaceInteractionTerm(coefsKQ, coefsK2, coefsK2Q, coefsK, motifFT);
            X = 0;

        }
        else if (mode == "reciprocalspace"){
            arma::rowvec k2 = system->kpoints.row(ki_index);
            recpotptr potential = selectReciprocalPotential(this->potential_); // Not sure at this point the effect of the chosen potential on this calculation
            D = reciprocalInteractionTerm(coefsK, coefsK2, coefsKQ, coefsK2Q, k, k2, k, k2, this->potential_, this->nReciprocalVectors);
            X = 0;
        }
        
        W(i) = - (D - X);                
    };

    double delta = 2.0/targetExciton.system->nk; // Adjust delta depending on number of k points
    double rho = targetExciton.pairDensityOfStates(energy, delta);
    std::cout << "DoS value: " << rho << endl;
    double hbar = 6.582119624E-16; // Units are eV*s

    transitionRate = (ncell*system->a)*2*PI*std::norm(arma::dot(W, initialState))*rho/hbar;


    return transitionRate;
}

/**
 * Method to print information about the exciton.
 * @return void 
 */
void ExcitonTB::printInformation(){
    std::cout << std::left << std::setw(30) << "Number of cells: " << ncell << endl;
    std::cout << std::left << std::setw(30) << "Valence bands:";
    for (uint i = 0; i < valenceBands.n_elem; i++){
        std::cout << valenceBands(i) << " ";
    }
    std::cout << endl;

    std::cout << std::left << std::setw(30) << "Conduction bands: ";
    for (uint i = 0; i < conductionBands.n_elem; i++){
        std::cout << conductionBands(i) << " ";
    }
    std::cout << "\n" << endl;

    std::cout << std::left << std::setw(30) << "Gauge used: " << gauge << std::endl;
    std::cout << std::left << std::setw(30) << "Calculation mode: " << mode << std::endl;
    if(mode == "reciprocalspace"){
        std::cout << std::left << std::setw(30) << "Gc: " << Gc_exciton  << " (angstrom^-1)" << std::endl;
        std::cout << std::left << std::setw(30) << "nG: " << nReciprocalVectors << std::endl;
    }
    std::cout << std::left << std::setw(30) << "Potential: " << potential_ << std::endl;
    if(exchange){
        std::cout << std::left << std::setw(30) << "Exchange: " << (exchange ? "True" : "False") << std::endl;
        std::cout << std::left << std::setw(30) << "Exchange potential: " << exchangePotential_ << std::endl;
    }
    std::cout << std::left << std::setw(30) << "Dielectric constant of embedding medium: " << eps_m << std::endl;
    std::cout << std::left << std::setw(30) << "Dielectric constant of substrate: " << eps_s << std::endl;
    if(arma::norm(Q) > 1E-7){
        std::cout << std::left << std::setw(30) << "Q: "; 
        for (auto qi : Q){
            std::cout << qi << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::left << std::setw(30) << "Scissor cut: " << scissor_ << std::endl;

    if (this->isscreeningset == true){
        std::cout << std::left << std::setw(40) << "\nNumber of valence bands included: " << this->nvalencebands_ << std::endl;
        std::cout << std::left << std::setw(40) << "Number of conduction bands included: " << this->nconductionbands_ << std::endl;
        std::cout << std::left << std::setw(40) << "Gcutoff: " << this->Gcutoff_ << std::endl;
        if (this->function_ == "dielectric" || this->function_ == "polarizability"){
            if (this->mode == "reciprocalspace"){
                arma::rowvec vector(this->trunreciprocalLattice_.row(this->Gs_(0)));
                arma::rowvec vectorprime(this->trunreciprocalLattice_.row(this->Gs_(1)));

                std::cout << std::left << std::setw(0) << "Selected G(" << std::setw(0) << this->Gs_(0) << "): " << vector(0) << " " << vector(1) << " " << vector(2) << std::endl;
                std::cout << std::left << std::setw(0) << "Selected G'(" << std::setw(0) << this->Gs_(1) << "): " << vectorprime(0) << " " << vectorprime(1) << " " << vectorprime(2) << std::endl;
            }

            if (this->mode == "realspace"){
                arma::rowvec vector(this->trunLattice_.row(this->Gs_(0)));
                arma::rowvec vectorprime(this->trunLattice_.row(this->Gs_(1)));

                std::cout << std::left << std::setw(0) << "Selected R(" << std::setw(0) << this->Gs_(0) << "): " << vector(0) << " " << vector(1) << " " << vector(2) << std::endl;
                std::cout << std::left << std::setw(0) << "Selected R'(" << std::setw(0) << this->Gs_(1) << "): " << vectorprime(0) << " " << vectorprime(1) << " " << vectorprime(2) << std::endl;
                std::cout << std::left << std::setw(0) << "Selected t_i = " << std::setw(0) << system->motif.row(ts(0)).subvec(0,2) << std::endl;
                std::cout << std::left << std::setw(0) << "Selected t_j = " << std::setw(0) << system->motif.row(ts(1)).subvec(0,2) << std::endl;
            }
           
        } else if (this->function_ == "none" || this->function_ == "inversedielectric") {
            std::cout << std::left << std::setw(40) << "Number of reciprocal lattice vectors: " << this->nGs << std::endl;
        }
    }
}

/**
 * Method to print reciprocal lattice
 * @return void 
 */
void ExcitonTB::printReciprocalLattice() {
    double cutoff = this->Gcutoff_;
    uint nGs_to_print = this->trunreciprocalLattice_.n_rows;
    uint nGs_total = this->trunreciprocalLattice_.n_rows;

    std::cout << "Reciprocal lattice vectors within cutoff |G| <= " << cutoff << " (angstrom^-1):" << std::endl;

    for (uint i = 0; i < nGs_total; i++) 
    {
        arma::rowvec G = this->trunreciprocalLattice_.row(i);

        if (arma::norm(G) <= cutoff) {
            std::cout << "G = G(" << i << ") = (" << G(0) << ", " << G(1) << ", " << G(2) << ") |G| = " << arma::norm(G) << std::endl;
        } else {
            return;
        }
    }
}

/**
 * Method to write the BZ mesh to a file.
 * @return void
*/
void ExcitonTB::writeBZtofile() const {
    std::string filename_k_points = "kgrid_" + std::to_string(this->ncell) + ".dat";

    std::ofstream k_points_file; 

    k_points_file.open(filename_k_points);

    if (!k_points_file.is_open()) { // check if the file was opened successfully
        std::cerr << "Error opening file\n";
        std::cerr << errno << "\n";
    }


    for(unsigned int i = 0; i < this->ncell*this->ncell; i++){
        auto k = this->system->kpoints.row(i);
        k_points_file << k(0) << " " << k(1) << " " << k(2) << std::endl;
    }

    k_points_file.close();
}

/* Method to print information of the inverse of the dielectric matrix into a file.
 * @return void 
 */
void ExcitonTB::writeInverseDielectricMatrix(std::string filename_dielectric) {

    FILE* textfile = fopen(filename_dielectric.c_str(), "w");

    if (textfile == NULL){
        std::cout << "File for inverse of the dielectric matrix failed to open. Exiting" << std::endl;
        exit(0);
    }

    std::cout << "Writing inverse of dielectric matrix fo file: " << filename_dielectric << std::endl;

    if (this->mode == "realspace"){
        std::cout << "Real space implementation ongoing. Terminating." << std::endl;
        exit(0);
    }

    if (this->mode == "reciprocalspace"){

        uint ngs = this->epsilonmatrix_.slice(0).n_rows;         // The number of G vectors can in general be different from the number of generated G vectors
        uint nqs = this->epsilonmatrix_.n_slices; // The number of q points can in general be different from the size of the BZ mesh 

        if (ngs == 0 || nqs == 0){
            std::cout << "Dielectric matrix was not computed. Exiting." << std::endl;
            exit(0);
        }

        if (this->Invepsilonmatrix_.is_empty()) {

            std::cout << "\nInverse dielectric matrix was not computed." << std::endl;

            try{
                this->Invepsilonmatrix_ = arma::cx_cube(ngs,ngs,nqs,arma::fill::zeros);

                this->invertDielectricMatrix();

            } catch(const std::exception& e){
                std::cerr << e.what() << std::endl;
            }
        }

        
        for(uint i = 0; i < nqs; i++){
            for (uint g = 0; g < ngs; g++){
                for (uint g2 = 0; g2 < ngs; g2++){
                    std::complex<double> aux = this->Invepsilonmatrix_.slice(i).at(g,g2);
                    fprintf(textfile, "%11.7lf%11.7lf", real(aux), imag(aux));
                }
                fprintf(textfile, "\n");
            }
        }

        fprintf(textfile, "%11.7lf %11.7lf %11.7lf %11.7lf %11.7lf", this->q0_(0), this->q0_(1), this->q0_(2), this->slope_, this->slope_perp_);
    }
    
    fclose(textfile);
}

/* Method to print information of the inverse of the dielectric matrix into a file.
 * @return void
 */
void ExcitonTB::writeDielectricMatrix(std::string filename_dielectric) const
{

    FILE *textfile = fopen(filename_dielectric.c_str(), "w");

    if (textfile == NULL)
    {
        std::cout << "File for the dielectric matrix failed to open. Exiting" << std::endl;
        exit(0);
    }

    std::cout << "Writing inverse of dielectric matrix fo file: " << filename_dielectric << std::endl;

    if (this->mode == "realspace")
    {
        std::cout << "Real space implementation ongoing. Terminating." << std::endl;
        exit(0);
    }

    if (this->mode == "reciprocalspace")
    {
        uint ngs = this->epsilonmatrix_.slice(0).n_rows; // The number of G vectors can in general be different from the number of generated G vectors
        uint nqs = this->epsilonmatrix_.n_slices;         // The number of q points can in general be different from the size of the BZ mesh

        for (unsigned int i = 0; i < nqs; i++)
        {
            for (unsigned int g = 0; g < ngs; g++)
            {
                for (unsigned int g2 = 0; g2 < ngs; g2++)
                {
                    std::complex<double> aux = this->epsilonmatrix_.slice(i).at(g, g2);
                    fprintf(textfile, "%11.7lf%11.7lf", real(aux), imag(aux));
                }
                fprintf(textfile, "\n");
            }
        }
    }

    fclose(textfile);
}

/* Method to print information of the polarizability matrix into a file.
 * @return void 
 */
void ExcitonTB::writePolarizabilityMatrix(std::string filename_dielectric) const {

    FILE* textfile = fopen(filename_dielectric.c_str(), "w");

    if (textfile == NULL){
        std::cout << "File for polarizability matrix failed to open. Exiting." << std::endl;
        exit(0);
    }

    std::cout << "Writing polarizability matrix fo file: " << filename_dielectric << std::endl;

    if (this->mode == "realspace"){
        std::cout << "Real space implementation ongoing. Terminating." << std::endl;
        exit(0);
    }

    if (this->mode == "reciprocalspace"){
        uint ngs = this->trunreciprocalLattice_.n_rows; // The number of G vectors can in general be different from the number of generated G vectors
        uint nqs = this->Chimatrix_.n_slices; // The number of q points can in general be different from the size of the BZ mesh 
        std::cout << "ngs = " << ngs << std::endl;
        for(unsigned int i = 0; i < nqs; i++){
            for (unsigned int g = 0; g < ngs; g++){
                for (unsigned int g2 = 0; g2 < ngs; g2++){
                    std::complex<double> aux = this->Chimatrix_.slice(i).at(g,g2);
                    fprintf(textfile, "%11.7lf%11.7lf", real(aux), imag(aux));
                }
                fprintf(textfile, "\n");
            }
        }
        std::cout << "Polarizability_00 =  " << this->Chimatrix_.slice(0).at(0,0) << std::endl;
    }
    
    fclose(textfile);
}

/* Method to read the inverse of the dielectric matrix from a pre-existent file.
 * @return void 
 */
void ExcitonTB::readInverseDielectricMatrix(std::string filename_screening) {

    std::complex<double> imag(0,1);
    
    std::ifstream file;

    file.open(filename_screening);

    if (!file.is_open()){
        std::cout << "File for inverse of the dielectric matrix failed to open or does not exist. Terminating." << std::endl;
        exit(0);
    }

    std::cout << "Reading inverse of dielectric matrix from file: " << filename_screening << std::endl;

    if (this->mode == "realspace"){
        uint n_atoms = this->system->motif.n_rows;
        uint n_R_vectors = this->system->bravaisLattice.n_rows;
        uint n_positions = n_R_vectors*n_atoms - 1;

        std::cout << "Exciton computed with real space screening not implemented yet. Terminating." << std::endl;

        exit(0);
    }

    if (this->mode == "reciprocalspace"){

        uint ngs = this->trunreciprocalLattice_.n_rows;
        int nqs = system->nk;
        int line_counter = 0;
        uint column_counter = 0;
        int k_counter = 0;
        std::string line;

        std::string fl;
        std::getline(file, fl); // get first line

        std::istringstream iss(fl);
        double m_element;
        while(iss >> m_element) {
            column_counter++;
        }

        if (ngs != column_counter/2){
            std::cout << "The number of reciprocal vectors read from file and from the configuration do not coincide! Terminating." << std::endl;
            exit(0); 
        }

        file.seekg(0); // go back to the beginning of the file

        while (std::getline(file, line)) {
            ++line_counter;
        }

        file.clear(); // clear the error state

        int total_lines = line_counter;

        int read_nqs = (line_counter-1)/ngs; // Last line stores info for W_00(0) regularization

        if (nqs != read_nqs) {
            std::cout << "The number of k points read from data file  (" + std::to_string(read_nqs) + ")  and from the exciton configuration file (" + std::to_string(nqs) + ") do not coincide!\n The exciton can not be computed with the provided screening data file. Terminating." << std::endl;
            exit(0); 
        }

        file.seekg(0);

        if (this->Invepsilonmatrix_.is_empty()) {
            this->Invepsilonmatrix_ = arma::cx_cube(ngs,ngs,nqs,arma::fill::zeros);
        } else {
        
            this->Invepsilonmatrix_.reshape(ngs,ngs,nqs);
        }

        line_counter = 0;
        while (std::getline(file, line)) {
            std::istringstream ss(line);
            double Re_part = 0;
            double Im_part = 0;
            double aux;
            column_counter = 0;
            int pair_counter = 0;

            if (line_counter + 1 == total_lines) {
                column_counter = 0;
                while(ss >> aux) {   
                    if (column_counter == 0) {
                        this->q0_(0) = aux;
                    } else if (column_counter == 1) {
                        this->q0_(1) = aux;
                    } else if (column_counter == 2) {
                        this->q0_(2) = aux;
                    } else if (column_counter == 3) {
                        this->slope_ = aux;
                    } else if (column_counter == 4) {
                        this->slope_perp_= aux;
                    }
                    column_counter++;
                }
                break;
            }

            while(ss >> aux) {   
                if (column_counter%2 == 0) {
                    Re_part = aux;
                } else {
                    Im_part = aux;
                    this->Invepsilonmatrix_.slice(k_counter)(line_counter%ngs, (column_counter - 1)/2) = Re_part + imag*Im_part;
                }
                column_counter++;
            }

            //std::cout << "\n";
            ++line_counter;
            if (line_counter%ngs == 0){
                k_counter++;
            } 
        }

        double q0_norm = arma::norm(this->q0_);
        this->W00_at_0_ = (2 + 0.5*(this->slope_ + this->slope_perp_)*q0_norm) * ec * 1E10 / (2 * eps0 * q0_norm * system->unitCellArea);
    }

    std::cout << "Inverse of dielectric matrix read from file with success." << std::endl;

    file.close();
}
}