#include <math.h>
#include <iomanip>
#include "xatu/ExcitonTB.hpp"
#include "xatu/utils.hpp"
#include "xatu/davidson.hpp"

using namespace arma;
using namespace std::chrono;

/** 
 *Method to sort rows in a matrix by norm
 *@return void 
*/
void sortVectors(arma::mat& vectors){

    std::vector<arma::rowvec> vecs;

    for (int i = 0; i < (int)vectors.n_rows; ++i)
        vecs.push_back(vectors.row(i));

    std::sort(vecs.begin(), vecs.end(), [](const arma::rowvec & a, const arma::rowvec & b){ return arma::norm(a) < arma::norm(b); });

    for (int i = 0; i < (int)vectors.n_rows; ++i){
        vectors.row(i)(0) = vecs[i][0];
        vectors.row(i)(1) = vecs[i][1];
        vectors.row(i)(2) = vecs[i][2];
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
    this->ncell_      = ncell;
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

    if (2*nbands > system->basisdim){
        std::cout << "Error: Number of bands cannot be higher than actual material bands" << std::endl;
        exit(1);
    }

    if (bands.empty()){
        bands = arma::regspace<arma::ivec>(- nbands + 1, nbands);
    }

    initializeExcitonAttributes(ncell, bands, parameters, Q);

    std::vector<arma::s64> valence, conduction;
    for(int i = 0; i < bands.n_elem; i++){
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
    this->nReciprocalVectors_ = cfg.excitonInfo.nReciprocalVectors;
    this->regularization_ = cfg.excitonInfo.regularization;
    if (regularization_ == 0){
        regularization_ = system->a;
    }
    this->potential_ = cfg.excitonInfo.potential;
    this->exchangePotential_ = cfg.excitonInfo.exchangePotential;
}

/**
 * Method to generate the reciprocal lattice vectors.
 * @details Every reciprocal lattice vectors is G = l G1 + k G2, where (l,k) is a pair of signed integers
 * @param nreciprocal number of reciprocal vectors in each direction.
 * @return matrix with the list of reciprocal lattive vectors organized in rows.
*/

arma::mat ExcitonTB::generateReciprocalVectors(int nreciprocal){

    int nls = 2*nreciprocal + 1;
    int nks = 2*nreciprocal + 1;
    int ncombinations = nls*nks;
    arma::mat ReciprocalVectors(ncombinations,3,arma::fill::zeros);
    arma::imat listl(nls,1,arma::fill::zeros);
    arma::imat listk(nks,1,arma::fill::zeros);
    //arma::imat combinations(ncombinations,2,arma::fill::zeros);
    std:: vector<std::array<int, 2>> combinations;

    arma::rowvec G1 = system_->reciprocalLattice.row(0);
    arma::rowvec G2 = system_->reciprocalLattice.row(1);

    for (int i = 1; i <= nreciprocal; ++i){
        listl(2*i - 1) = -i;
        listl(2*i) = i;
    }

    for (int i = 1; i <= nreciprocal; ++i){
        listk(2*i - 1) = -i;
        listk(2*i) = i;
    }

    combinations.push_back({0,0});

    for (int i = 1; i <= nreciprocal; i=i+1){
        for (int j = 0; j <= i; ++j){

            if (j == 0){
                combinations.push_back({-i,j});
                combinations.push_back({i,j});
                combinations.push_back({j,-i});
                combinations.push_back({j,i});
            }
            

            if (j != 0 && j != i){
                combinations.push_back({-i,j});
                combinations.push_back({i,-j});
                combinations.push_back({j,-i});
                combinations.push_back({-j,i});
                combinations.push_back({-j,-i});
                combinations.push_back({j,i});
                combinations.push_back({-i,-j});
                combinations.push_back({i,j});
            }
            
            if (j == i){
                combinations.push_back({-i,j});
                combinations.push_back({i,-j});
                combinations.push_back({-j,-i});
                combinations.push_back({j,i});
            }
        }
    }

    if (ncombinations != combinations.size()){
        std::cout << "Length of combinations vector different from the number of combinations" << std::endl;
        exit(0);
    }

    for (int i = 0; i < ncombinations; ++i){
        int l = combinations[i][0];
        int k = combinations[i][1];
        ReciprocalVectors.row(i) = l*G1 + k*G2;
    }

    return ReciprocalVectors;
}
/**
 * Method to set the screening attributes of an exciton object from a ScreeningConfiguration object.
 * @details Overload of the method to use a configuration object. Based on the parametric method.
 * @param cfg ScreeningConfiguration object from parsed file.
 * @return void
 */
void ExcitonTB::initializeScreeningAttributes(const ScreeningConfiguration& cfg){
    int nvalencebands        = cfg.screeningInfo.nvbands;
    int nconductionbands     = cfg.screeningInfo.ncbands;
    int nremovedbands        = cfg.screeningInfo.nrmcbands;
    arma::rowvec q           = cfg.screeningInfo.q;
    arma::ivec gs            = cfg.screeningInfo.Gs;
    std::string function     = cfg.screeningInfo.function;

    this->isscreeningset = true;
    this->function_ = function;

    int totalvbands = system->fermiLevel + 1;
    int totalcbands = system->basisdim - totalvbands;

    if (nvalencebands > totalvbands  || nconductionbands > totalcbands){
        std::cout << "Number of included bands cannot be higher than the number of material bands" << std::endl;
        std::cout << "Number of total valence bands = " << totalvbands<< std::endl;
        std::cout << "Number of total conduction bands = " << totalcbands << std::endl;
    
        exit(1);
    }

    if (nvalencebands + nconductionbands > system->basisdim){
        std::cout << "Error: Number of bands cannot be higher than actual material" << std::endl;
        std::cout << "Total number of bands is " << system->basisdim << std::endl;
        exit(1);
    }

    this->q_ = q;

    this->nvalencebands_ = nvalencebands;
    this->nconductionbands_ = nconductionbands;
    this->nrmcbands_ = nremovedbands;

    this->Gs_ = gs;

    std::vector<arma::s64> valence, conduction;
    int basisdim = system->basisdim - nremovedbands;

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

    if (cfg.screeningInfo.Gcutoff_found){
        this->Gcutoff_ = cfg.screeningInfo.Gcutoff;
    } else {
        this->Gcutoff_ = (this->ncell)/2.5;
    }

    double radius = this->Gcutoff_ * arma::norm(system->reciprocalLattice.row(0));
    this->trunreciprocalLattice_ = system_->truncateReciprocalSupercell(this->nReciprocalVectors, radius);
    
    int ngs  = this->nGs = this->trunreciprocalLattice_.n_rows;

    if (gs(0) >= ngs || gs(1) >= ngs){
        std::cout << "Error: Index of the reciprocal vector must not be higher than number of reciprocal vectors" << std::endl;
        std::cout << "Number of reciprocal vectors is " << ngs << std::endl;

        exit(1);
    }

    
    // std::cout << "this->trunreciprocalLattice_.n_rows = " << this->trunreciprocalLattice_.n_rows << std::endl;
    // std::cout << "||system->reciprocalLattice.row(0)|| = " << arma::norm(system->reciprocalLattice.row(0)) << std::endl;
    // std::cout << "radius as of Gcutoff = " << radius << std::endl;
    // std::cout << "radius as of cutoff =  = " << cutoff_ * arma::norm(system->reciprocalLattice.row(0)) << std::endl;

    if (cfg.screeningInfo.function == "none"){
        int nks = this->ncell*this->ncell;
        this->Chimatrix_ = arma::cx_cube(ngs, ngs, nks, arma::fill::zeros);
        this->epsilonmatrix_ = arma::cx_cube(ngs, ngs, nks, arma::fill::zeros);
        this->Invepsilonmatrix_ = arma::cx_cube(ngs, ngs, nks, arma::fill::zeros);
    }
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
        cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
        exit(1);
    }

    // arma::ivec is implemented with typedef s64
    std::vector<arma::s64> valence, conduction;
    for(int i = 0; i < bands.n_elem; i++){
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
        cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
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
    initializeScreeningAttributes(screeningConfig);
}

ExcitonTB::ExcitonTB(std::shared_ptr<SystemTB> sys, int ncell, const arma::ivec& bands, 
                     const arma::rowvec& parameters, const arma::rowvec& Q){

    
    system_ = sys;
    initializeExcitonAttributes(ncell, bands, parameters, Q);

    if (bands.n_elem > system->basisdim){
        std::cout << "Error: Number of bands cannot be higher than actual material bands" << std::endl;
        exit(1);
    }

    // arma::ivec is implemented with typedef s64
    std::vector<arma::s64> valence, conduction;
    for(int i = 0; i < bands.n_elem; i++){
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
        cout << "Error: Number of bands cannot be higher than actual material bands" << endl;
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
 * @param nReciprocalVector Number of reciprocal vectors to sum over.
 * @return void 
 */
void ExcitonTB::setReciprocalVectors(int nReciprocalVectors){
    if(nReciprocalVectors < 0){
        throw std::invalid_argument("setReciprocalVectors(): given number must be positive");
    }
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

/*---------------------------------------- Potentials ----------------------------------------*/

/** 
 * Purpose: Compute Struve function H0(x).
 * Source: http://jean-pierre.moreau.pagesperso-orange.fr/Cplus/mstvh0_cpp.txt 
 * @param X x --- Argument of H0(x) ( x ò 0 )
 * @param SH0 SH0 --- H0(x). The return value is written to the direction of the pointer.
*/
void ExcitonTB::STVH0(double X, double *SH0) {
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
 * Calculate value of interaction potential (Keldysh). Units are eV.
 * @details If the distance is zero, then the interaction is renormalized to be V(a) since
 * V(0) is infinite, where a is the lattice parameter. Also, for r > cutoff the interaction is taken to be zero.
 * @param r Distance at which we evaluate the potential.
 * @return Value of Keldysh potential, V(r).
 */
double ExcitonTB::keldysh(double r){
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
double ExcitonTB::coulomb(double r){
    double cutoff = arma::norm(system->bravaisLattice.row(0)) * cutoff_ + 1E-5;
    r = abs(r);
    if (r > cutoff){
        return 0.0;
    }
    return (r != 0) ? ec/(4E-10*PI*eps0*r) : ec*1E10/(4*PI*eps0*regularization);    
}

/**
 * RPA potential in real space.
 * Implementation in progress, returns the same as keldysh.
 */
double ExcitonTB::rpa(double r){
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

potptr ExcitonTB::selectPotential(std::string potential){
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



/*---------------------------------------- Fourier transforms ----------------------------------------*/
/**
 * Evaluates the Fourier transform of the Keldysh potential, which is an analytical expression.
 * @param q kpoint where we evaluate the FT.
 * @return Fourier transform of the potential at q, FT[V](q).
 */
double ExcitonTB::coulombFT(arma::rowvec q){
    double radius = cutoff*arma::norm(system->reciprocalLattice.row(0));
    double potential = 0;
    double eps = arma::norm(system->reciprocalLattice.row(0))/totalCells;

    double qnorm = arma::norm(q);
    if (qnorm < eps){
        potential = 0;
    }
    else{
        potential = 1/(qnorm);
    }
    
    potential = potential*ec*1E10/(2*eps0);
    
    return potential;
}  

/**
 * Evaluates the Fourier transform of the Keldysh potential, which is an analytical expression.
 * @param q kpoint where we evaluate the FT.
 * @return Fourier transform of the potential at q, FT[V](q).
 */
double ExcitonTB::keldyshFT(arma::rowvec q){
    double radius = cutoff*arma::norm(system->reciprocalLattice.row(0));
    double potential = 0;
    double eps_bar = (eps_m + eps_s)/2;
    double eps = arma::norm(system->reciprocalLattice.row(0))/totalCells;

    double qnorm = arma::norm(q);
    if (qnorm < eps){
        potential = 0;
    }
    else{
        potential = 1/(qnorm*(1 + r0*qnorm));
    }
    
    potential = potential*ec*1E10/(2*eps0*eps_bar*system->unitCellArea*totalCells);
    return potential;
}

/**
 * Evaluates the RPA potential in reciprocal space.
 * Implementation in progress, returns the same as keldysh.
 */
double ExcitonTB::rpaFT(arma::rowvec q){
    double radius = cutoff*arma::norm(system->reciprocalLattice.row(0));
    double potential = 0;
    double eps_bar = (eps_m + eps_s)/2;
    double eps = arma::norm(system->reciprocalLattice.row(0))/totalCells;

    double qnorm = arma::norm(q);
    if (qnorm < eps){
        potential = 0;
    }
    else{
        potential = 1/(qnorm*(1 + r0*qnorm));
    }
    
    potential = potential*ec*1E10/(2*eps0*eps_bar*system->unitCellArea*totalCells);
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

    for(int n = 0; n < cells.n_rows; n++){
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
                                     int nrcells){
    
    std::complex<double> Ic, Iv;
    std::complex<double> term = 0;
    double radius = cutoff * arma::norm(system->reciprocalLattice.row(0));
    arma::mat reciprocalVectors = system_->truncateReciprocalSupercell(nrcells, radius);

    for(int i = 0; i < reciprocalVectors.n_rows; i++){
        auto G = reciprocalVectors.row(i);

        Ic = blochCoherenceFactor(coefsKQ, coefsK2Q, kQ, k2Q, G);
        Iv = blochCoherenceFactor(coefsK, coefsK2, k, k2, G);

        term += Ic*conj(Iv)*keldyshFT(k - k2 + G);
    }

    return term;
};

/**
 * Calculation of Bloch coherence factors, required to compute the interaction terms in reciprocal space.
 * @param coefs1 Vector of eigenstate |n,k>.
 * @param coefs2 Vector of eigenstate |n',k'>.
 * @param k1 kpoint k.
 * @param k2 kpoint k'.
 * @param G Reciprocal lattice vector used to compute the coherence factor.
 * @return Bloch coherence factor I evaluated at G for states |nk>, |n'k'>.
 */
std::complex<double> ExcitonTB::blochCoherenceFactor(const arma::cx_vec& coefs1, const arma::cx_vec& coefs2,
                                                    const arma::rowvec& k1, const arma::rowvec& k2,
                                                    const arma::rowvec& G){

    std::complex<double> imag(0, 1);
    arma::cx_vec coefs = arma::conj(coefs1) % coefs2;
    arma::cx_vec phases = arma::ones<arma::cx_vec>(system->basisdim);

    int index_min = 0;
    int index_max = -1;

    for(int i = 0; i < system->natoms; i++){
        int species = system->motif.row(i)(3);
        arma::rowvec atomPosition = system->motif.row(i).subvec(0, 2); 

        index_max += system->orbitals(species);
        phases.subvec(index_min, index_max) *= exp(imag*arma::dot(k1 - k2 + G, atomPosition));

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
    for(int i = 0; i < bands.n_elem; i++){
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
    for(int j = 0; j < sums.n_elem; j++){
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

    int nk = system->nk;
    int natoms = system->natoms;
    int basisdim = system->basisdim;

    this->eigvecKStack_  = arma::cx_cube(basisdim, nTotalBands, nk);
    this->eigvecKQStack_ = arma::cx_cube(basisdim, nTotalBands, nk);
    this->eigvalKStack_  = arma::mat(nTotalBands, nk);
    this->eigvalKQStack_ = arma::mat(nTotalBands, nk);
    this->ftMotifStack   = arma::cx_cube(natoms, natoms, system->meshBZ.n_rows);
    this->ftMotifQ       = arma::cx_mat(natoms, natoms);

    if (this->isscreeningset == true){
        this->eigveckStack_  = arma::cx_cube(basisdim, basisdim, nk); // For the dielectric function
        this->eigvalkStack_  = arma::mat(basisdim, nk);
    }

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
    for (int i = 0; i < nk; i++){
        arma::rowvec k = system->kpoints.row(i);
        system->solveBands(k, auxEigVal, auxEigvec);

        auxEigvec = fixGlobalPhase(auxEigvec);
        eigvalKStack_.col(i) = auxEigVal(bandList);
        eigvecKStack_.slice(i) = auxEigvec.cols(bandList);

        if (this->isscreeningset == true){
            eigvalkStack_.col(i) = auxEigVal;
            eigveckStack_.slice(i) = auxEigvec;
        }
        
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
                //     cout << "\r" << "[" << std::string(percent / 5, '|') << std::string(100 / 5 - percent / 5, ' ') << "]";
                //     cout << percent << "%";
                //     std::cout.flush();
                //     displayNext += step;
                // }
        }
        std::cout << "Done\n" << std::endl;
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

    std::cout << "Initializing basis for BSE... " << std::flush;
    initializeBasis();
    generateBandDictionary();

    initializeResultsH0();
}

/*------------------------------------ Static dielectric function matrix elements ------------------------------------*/
/**
 * Method to compute the matrix element of the static polarizability at the specified pair the (R + t_i, R' + t_j) from an input file.
 * @details Writes the polarizability in the file polarizability_convergence.dat as a 
 * function of the number of included conduction bands
 * @param R 1st Lattice vector
 * @param R2 2nd Lattice vector
 * @param t_i 1st atom of the motif
 * @param t_j 2nd atom of the motif
 * @return Polarizability
*/
std::complex<double> ExcitonTB::computesinglePolarizability(const arma::rowvec& R, const arma::rowvec& R2, const int i, const int j) {


    std::ofstream polarfile("../examples/screeningconfig/realspace_polarizability_convergence.dat"); 

    if (!polarfile.is_open()) { // check if the file was opened successfully
        std::cerr << "Error opening file\n";
    }

    std::complex<double> I(0, 1);

    int nk = system->nk;
    int natoms = system->natoms;
    int basisdim = system->basisdim;

    int nvbands = valencebands.size();
    int ncbands = conductionbands.size();

    int nvbandsincluded = this->nvalencebands_;
    int ncbandsincluded = this->nconductionbands_;

    int upperindexcband = nvbands + ncbandsincluded - 1;
    int lowerindexvbands = nvbands - nvbandsincluded;

    std::cout << "Total of valence bands = " << nvbands << "\n";
    std::cout << "Total of conduction bands = " << ncbands << "\n";
    std::cout << "fermi level = " << system->fermiLevel << "\n";

    std::complex<double> term = 0.;

    this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
    this->ftMotifStack   = arma::cx_cube(natoms, natoms, system->meshBZ.n_rows);
    this->ftMotifQ       = arma::cx_mat(natoms, natoms);

    vec auxEigVal(basisdim);
    arma::cx_mat auxEigvec(basisdim, basisdim);

    
    arma::cx_vec coefsk, coefsk2;

    int i_index = 0;
    int j_index = 0;
    
    for(unsigned int atom_index = 0; atom_index < i; atom_index++){
        int norbitals = system->orbitals(system->motif.row(atom_index)(3));

        i_index += norbitals;
    }

    for(unsigned int atom_index = 0; atom_index < j; atom_index++){
        int norbitals = system->orbitals(system->motif.row(atom_index)(3));

        j_index += norbitals;
    }

    arma::vec t_i = system->motif.row(i);
    arma::vec t_j = system->motif.row(j);

    int norbitals_alpha = system->orbitals(system->motif.row(i)(3));
    int norbitals_beta = system->orbitals(system->motif.row(j)(3));

    for (int ic = nvbands; ic <= upperindexcband; ic++){
    
        for (int iv = nvbands - 1; iv >= nvbands - nvbandsincluded; iv--){

            for (int ik = 0; ik < nk; ik++){

                arma::rowvec k = system->kpoints.row(ik);

                // Using the atomic gauge
                if(gauge == "atomic"){
                    coefsk = system_->latticeToAtomicGauge(eigveckStack_.slice(ik).col(iv), system->kpoints.row(ik));
                }
                else{                            
                    coefsk = eigveckStack_.slice(ik).col(iv);
                }
                
                for (int ik2 = 0; ik2 < nk; ik2++){

                    arma::rowvec k2 = system->kpoints.row(ik2);

                    // Using the atomic gauge
                    if(gauge == "atomic"){
                        coefsk2 = system_->latticeToAtomicGauge(eigveckStack_.slice(ik2).col(ic), system->kpoints.row(ik));
                    }
                    else{                            
                        coefsk2 = eigveckStack_.slice(ik2).col(ic);
                    }

                    arma::cx_vec CoefArray =  arma::conj(coefsk) % coefsk2;

                    std::complex<double> sum_alpha, sum_beta = 0;

                    sum_alpha = arma::sum(CoefArray.subvec(i_index, i_index + norbitals_alpha - 1));
                    sum_beta = arma::sum(CoefArray.subvec(j_index, j_index + norbitals_beta - 1));

                    term += 2*real(sum_alpha*sum_beta*exp(I*arma::dot(k - k2, R - R2))) / (eigvalkStack_.col(ik2)(ic) - eigvalkStack_.col(ik)(iv));

                    //std::cout << "ik = " << k << ", iv = " << iv << ", ic = " << ic << "\n"; 
                
                }
            }
                polarfile << nvbands - iv << " " << ic - nvbands + 1 << " " << real(term)/(totalCells) << " " << imag(term)/(totalCells) << "\n";
        }
    }

    polarfile.close();

    return term/((std::complex<double>)totalCells);
}

/**
 * Method to compute the (G,G') matrix element of the static polarizability at the specified momentum vector q in the input file.
 * @details Writes the polarizability in the file polarizability_convergence.dat as a 
 * function of the number of included conduction bands
 * @param q Momentum vector q specified in the input file
 * @return Polarizability
*/
std::complex<double> ExcitonTB::computesinglePolarizability(arma::rowvec& q) {

    if(mode == "realspace"){
        std::cout << "Real space dielectric function not implemented yet. Exiting." << std::endl;

        std::exit(0);
    }
    std::cout << "cutoff = " << cutoff << "\n";

    std::ofstream polarfile("../examples/screeningconfig/polarizability_convergence.dat"); 

    if (!polarfile.is_open()) { // check if the file was opened successfully
        std::cerr << "Error opening file\n";
    }

    int nk = system->nk;
    int natoms = system->natoms;
    int basisdim = system->basisdim;

    arma::rowvec g = this->trunreciprocalLattice_.row(this->Gs(0)); // Sets G
    arma::rowvec g2 = this->trunreciprocalLattice_.row(this->Gs(1)); // Sets G'

    int nvbands = valencebands.size();
    int ncbands = conductionbands.size();

    int nvbandsincluded = this->nvalencebands_;
    int ncbandsincluded = this->nconductionbands_;

    int upperindexcband = nvbands + ncbandsincluded - 1;
    int lowerindexvbands = nvbands - nvbandsincluded;

    std::cout << "Total of valence bands = " << nvbands << "\n";
    std::cout << "Total of conduction bands = " << ncbands << "\n";
    std::cout << "fermi level = " << system->fermiLevel << "\n";

    std::complex<double> term = 0.;

    this->eigveckqStack_ = arma::cx_cube(basisdim, basisdim, nk);
    this->eigvalkqStack_ = arma::mat(basisdim, nk);
    this->ftMotifStack   = arma::cx_cube(natoms, natoms, system->meshBZ.n_rows);
    this->ftMotifQ       = arma::cx_mat(natoms, natoms);

    vec auxEigVal(basisdim);
    arma::cx_mat auxEigvec(basisdim, basisdim);

    std::cout << "Diagonalizing H0 for all k+q points ... " << std::flush;

    for (int i = 0; i < nk; i++){
        arma::rowvec kq = system->kpoints.row(i) + q;
        system->solveBands(kq, auxEigVal, auxEigvec);

        auxEigvec = fixGlobalPhase(auxEigvec);
        eigvalkqStack_.col(i) = auxEigVal;
        eigveckqStack_.slice(i) = auxEigvec;
    };

    std::cout << "Done \n";
    
    if(mode == "realspace"){
        std::cout << "Real space dielectric function not implemented yet. Exiting." << std::endl;

        std::exit(0);

    } else if (mode == "reciprocalspace"){

        arma::cx_vec coefskq, coefsk;

        for (int ic = nvbands; ic <= upperindexcband; ic++){
        
            for (int iv = nvbands - 1; iv >= nvbands - nvbandsincluded; iv--){

                for (int ik = 0; ik < nk; ik++){

                    arma::rowvec k = system->kpoints.row(ik);
                    arma::rowvec kq = system->kpoints.row(ik) + q;
            
                    // Using the atomic gauge
                    if(gauge == "atomic"){
                        coefskq = system_->latticeToAtomicGauge(eigveckqStack_.slice(ik).col(iv), system->kpoints.row(ik));
                        coefsk = system_->latticeToAtomicGauge(eigveckStack_.slice(ik).col(ic), system->kpoints.row(ik));
                    }
                    else{                            
                        coefskq = eigveckqStack_.slice(ik).col(iv);
                        coefsk = eigveckStack_.slice(ik).col(ic);
                    }

                    std::complex<double> IvcG = blochCoherenceFactor(coefskq, coefsk, kq, k, g);
                    std::complex<double> IvcG2 = blochCoherenceFactor(coefskq, coefsk, kq, k, g2);

                    term += IvcG*std::conj(IvcG2) / (eigvalkqStack_.col(ik)(iv) - eigvalkStack_.col(ik)(ic));

                    //std::cout << "ik = " << k << ", iv = " << iv << ", ic = " << ic << "\n"; 
                }
                    polarfile << nvbands - iv << " " << ic - nvbands + 1 << " " << real(term)/(system->unitCellArea*totalCells) << " " << imag(term)/(system->unitCellArea*totalCells) << "\n";
            }
        }
    } else {
        std::cout << "Mode not recognized, must be 'realspace' or 'reciprocalspace'. Exiting." << std::endl;
        std::exit(0);
    }

    polarfile.close();

    for(int i = 0; i < this->trunreciprocalLattice_.n_rows; i++){
        auto G = this->trunreciprocalLattice_.row(i);
        std::cout << "G(" << i << ") = (" << G(0) << ", " << G(1) << ", " << G(2) << "), |G| = " << arma::norm(G) << std::endl;  
    }
    
    std::cout << "Selected (G,G') pair:" << "\n";
    std::cout << "G = G(" << this->Gs(0) << ") = (" << g(0) << ", " << g(1) << ", " << g(2) << ")" << std::endl;
    std::cout << "G' = G(" << this->Gs(1) << ") = (" << g2(0) << ", " << g2(1) << ", " << g2(2) << ")" << std::endl;

    return term/(system->unitCellArea*totalCells);
}

arma::mat generatecombinations(int nvbands, int basisdim, int nks){
    arma::mat vbands(1,nvbands,arma::fill::zeros);
    arma::mat cbands(1,basisdim-nvbands,arma::fill::zeros);
    arma::mat kindices(1,nks,arma::fill::zeros);

    arma::mat bands = arma::kron(vbands,cbands);
}

/**
 * Method to compute the (G,G') matrix element of the static polarizability at the specified momentum vector q.
 * @details The momentum q has to be specified through an index, matching a k point in the BZ mesh
 * @param G Reciprocal lattice vector G
 * @param G2 Reciprocal lattice vector G2
 * @param q Momentum vector q
 * @return Polarizability
*/
std::complex<double> ExcitonTB::reciprocalPolarizabilityMatrixElement(const arma::rowvec& G, const arma::rowvec& G2, int iq) {

    int nk = system->nk;
    int natoms = system->natoms;
    int basisdim = system->basisdim;

    int nvbands = valencebands.size();
    int ncbands = conductionbands.size();

    int nvbandsincluded = this->nvalencebands_;
    int ncbandsincluded = this->nconductionbands_;

    int upperindexcband = nvbands + ncbandsincluded - 1;
    int lowerindexvbands = nvbands - nvbandsincluded;

    arma::cx_vec coefskq, coefsk;

    std::complex<double> term = 0.;

    for (int ik = 0; ik < nk; ik++){

        arma::rowvec k = system->kpoints.row(ik);
        arma::rowvec kq = system->kpoints.row(ik) + system->kpoints.row(iq);

        int kqindex = system_->findEquivalentPointBZ(kq, ncell);

        arma::cx_dmat *auxk_slice = &eigveckStack_.slice(ik);
        arma::cx_dmat *auxkq_slice = &eigveckStack_.slice(kqindex);

        for (int ic = nvbands; ic <= upperindexcband; ic++){

            // Using the atomic gauge
            if(gauge == "atomic"){
                coefsk = system_->latticeToAtomicGauge(auxk_slice->col(ic), system->kpoints.row(ik));
            } else {                            
                coefsk = auxk_slice->col(ic);
            }
        
            for (int iv = nvbands - 1; iv >= nvbands - nvbandsincluded; iv--){

                // Using the atomic gauge
                if(gauge == "atomic"){
                    coefskq = system_->latticeToAtomicGauge(auxkq_slice->col(iv), system->kpoints.row(kqindex));
                } else {                            
                    coefskq = auxkq_slice->col(iv);
                }

                std::complex<double> IvcG = blochCoherenceFactor(coefskq, coefsk, kq, k, G);
                std::complex<double> IvcG2 = blochCoherenceFactor(coefskq, coefsk, kq, k, G2);

                term += IvcG*std::conj(IvcG2) / (eigvalkStack_.col(kqindex)(iv) - eigvalkStack_.col(ik)(ic));
            }
        }
    }

    return term/(system->unitCellArea*totalCells);
}

/**
 * Method to compute the (G,G') static polarizability in the BZ mesh.
 * @details Opens 'polarizability_mesh.dat' file and writes in it the values of the polarizability at each point in the BZ mesh
 * @return void
*/
void ExcitonTB::PolarizabilityMesh(){

    auto start = high_resolution_clock::now();

    std::cout << "Computing polarizability in the BZ mesh... \n" << std::flush;

    std::ofstream polarfile; 

    polarfile.open("../examples/screeningconfig/polarizability_mesh.dat");

    if (!polarfile.is_open()) { // check if the file was opened successfully
        std::cerr << "Error opening file\n";
        std::cerr << errno << "\n";
        
        exit(1);
    }

    int nq = system->nk;

    arma::cx_vec Chi(nq, arma::fill::zeros);

    double radius = cutoff * arma::norm(system->reciprocalLattice.row(0));
    arma::mat reciprocalVectors = system_->truncateReciprocalSupercell(this->nReciprocalVectors, radius);

    arma::rowvec g = reciprocalVectors.row(this->Gs(0)); // Sets G
    arma::rowvec g2 = reciprocalVectors.row(this->Gs(1)); // Sets G'

    #pragma omp parallel for
    for (int iq = 0; iq < nq; iq++){
        Chi(iq) = reciprocalPolarizabilityMatrixElement(g, g2, iq);
        std::cout << "iq = " << iq << "\n";
    }

    std::cout << "Chi computed" << std::endl;
    
    for (int iq = 0; iq < nq; iq++){
        auto q = system_->kpoints.row(iq);
        polarfile << q(0) << " " << q(1) << " " << q(2) << " " << real(Chi(iq)) << " " << imag(Chi(iq)) << "\n";
    }

    polarfile.close();

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Done in " << duration.count()/1000.0 << " s." << std::endl;
}

/**
 * Method to fetch the index of a reciprocal lattice vector.
 * @details returns -1 if not found.
 * @return int
*/
int ExcitonTB::fecthReciprocalLatticeVector(arma::rowvec G){

    int nGs = this->trunreciprocalLattice_.n_rows;

	for (int i = 0; i < nGs; ++i){
        if (arma::norm(G - this->trunreciprocalLattice_.row(i)) < 1E-7){
            return i;
        }
    }

    return -1; //G not found
}

/**
 * Method to compute the static polarizability matrix in the BZ mesh.
 * @return void
*/
void ExcitonTB::computeDielectricMatrix(){

    auto start = high_resolution_clock::now();

    std::cout << "Computing dielectric matrix in the BZ mesh... " << std::flush;

    arma::mat ReciprocalVectors = this->trunreciprocalLattice_;
    int nGs = ReciprocalVectors.n_rows;
    int Ncells = this->ncell_;
    int odd = Ncells%2;
    int nq = Ncells*(Ncells - odd)/2 + (Ncells - odd)/2 + 1; //Only half of the BZ
    int Nktotal = system->nk;
    nq = Nktotal;

    // std::ofstream dielectricfile("../examples/screeningconfig/inversedielectric" + std::to_string(nGs) + ".txt"); 


    // if (!dielectricfile.is_open()) { // check if the file was opened successfully
    //     std::cerr << "Error opening file\n";
    // }

    
    // std::cout << "cutoff = " << cutoff << std::endl;
    // std::cout << "Gcutoff = " << this->Gcutoff_ << std::endl;

    // arma::rowvec g = ReciprocalVectors.row(this->Gs_(0));
    // arma::rowvec g2 = ReciprocalVectors.row(this->Gs_(1));

    // std::cout << "Selected (G,G') pair:" << "\n";
    // std::cout << "G = G(" << this->Gs_(0) << ") = (" << g(0) << ", " << g(1) << ", " << g(2) << ")" << std::endl;
    // std::cout << "G' = G(" << this->Gs_(1) << ") = (" << g2(0) << ", " << g2(1) << ", " << g2(2) << ")" << std::endl;

    //int iq = system_->findEquivalentPointBZ(arma::rowvec(3,arma::fill::zeros), ncell);
    // iq = 210;
    // std::cout << "q = " << system->kpoints.row(iq)(0) << " " << system->kpoints.row(iq)(1) << " " << system->kpoints.row(iq)(2) << std::endl;
    
    arma::cx_mat auxvecsol(nGs,nGs,arma::fill::zeros);
    arma::cx_mat auxvec(nGs,nGs,arma::fill::eye);

    arma::imat indecesg(nGs*(nGs+1)/2,2,arma::fill::zeros);
    int i=0;

    arma::imat indecesqg(nq*nGs*(nGs+1)/2,3,arma::fill::zeros);

    // for (int g = 0; g < nGs; g++){

    //     for (int g2 = g; g2 < nGs; g2++){
    //         indecesg.row(i)(0) = g;
    //         indecesg.row(i)(1) = g2;
    //         i++;
    //     }
    // }

    for (int iq = 0; iq < nq; iq++){
        for (int g = 0; g < nGs; g++){
            for (int g2 = g; g2 < nGs; g2++){
                indecesqg.row(i)(0) = iq;
                indecesqg.row(i)(1) = g;
                indecesqg.row(i)(2) = g2;
                i++;
            }
        }
    }

    for (int i = 0; i < nGs; i++){

        arma::rowvec G = ReciprocalVectors.row(i);                

        std::cout << "G = G(" << i << ") = (" << G(0) << ", " << G(1) << ", " << G(2) << ") |G| = " << arma::norm(G) << std::endl;

    }

    // #pragma omp parallel for
    // for (int i = 0; i < nGs*(nGs+1)/2; i++){

    //     int g = indecesg.row(i)(0);
    //     int g2 = indecesg.row(i)(1);

    //     arma::rowvec G = ReciprocalVectors.row(g);                
    //     arma::rowvec G2 = ReciprocalVectors.row(g2);

    //     this->Chimatrix_.slice(iq).row(g)(g2) = reciprocalPolarizabilityMatrixElement(G, G2, iq);
    //     this->Chimatrix_.slice(iq).row(g2)(g) = std::conj(this->Chimatrix_.slice(iq).row(g)(g2));

    //     double potentialg = coulombFT(system->kpoints.row(iq) + G);
    //     double potentialg2 = coulombFT(system->kpoints.row(iq) + G2);
    //     double kroneckerdelta = g == g2? 1 : 0;

    //     this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg*this->Chimatrix_.slice(iq).row(g)(g2);
    //     this->epsilonmatrix_.slice(iq).row(g2)(g) = kroneckerdelta - potentialg2*this->Chimatrix_.slice(iq).row(g2)(g);
    // } 


    #pragma omp parallel for
    for (int i = 0; i < nq*nGs*(nGs+1)/2; i++){

        int iq = indecesqg.row(i)(0);
        int g  = indecesqg.row(i)(1);
        int g2 = indecesqg.row(i)(2);

        arma::rowvec G = ReciprocalVectors.row(g);                
        arma::rowvec G2 = ReciprocalVectors.row(g2);

        // int negativeG = fecthReciprocalLatticeVector(-G);
        // int negativeG2 = fecthReciprocalLatticeVector(-G2);

        this->Chimatrix_.slice(iq).row(g)(g2) = reciprocalPolarizabilityMatrixElement(G, G2, iq);
        this->Chimatrix_.slice(iq).row(g2)(g) = std::conj(this->Chimatrix_.slice(iq).row(g)(g2));

        // this->Chimatrix_.slice(Nktotal - iq - 1).row(negativeG2)(negativeG) = this->Chimatrix_.slice(iq).row(g)(g2);
        // this->Chimatrix_.slice(Nktotal - iq - 1).row(negativeG)(negativeG2) = this->Chimatrix_.slice(iq).row(g2)(g);

        double potentialg = coulombFT(system->kpoints.row(iq) + G);
        double potentialg2 = coulombFT(system->kpoints.row(iq) + G2);

        // double potentialnegativeG = coulombFT(system->kpoints.row(Nktotal - iq - 1) - G);
        // double potentialnegativeG2 = coulombFT(system->kpoints.row(Nktotal - iq - 1) - G2);
        double kroneckerdelta = g == g2? 1 : 0;

        this->epsilonmatrix_.slice(iq).row(g)(g2) = kroneckerdelta - potentialg*this->Chimatrix_.slice(iq).row(g)(g2);
        this->epsilonmatrix_.slice(iq).row(g2)(g) = kroneckerdelta - potentialg2*this->Chimatrix_.slice(iq).row(g2)(g);

        // this->epsilonmatrix_.slice(Nktotal - iq - 1).row(negativeG2)(negativeG) = kroneckerdelta - potentialnegativeG2*this->Chimatrix_.slice(Nktotal - iq - 1).row(negativeG2)(negativeG);
        // this->epsilonmatrix_.slice(Nktotal - iq - 1).row(negativeG)(negativeG2) = kroneckerdelta - potentialnegativeG*this->Chimatrix_.slice(Nktotal - iq - 1).row(negativeG)(negativeG2);

        //std::cout << i << ",";
    } 

    auto stop_dielectric_matrix_mesh = high_resolution_clock::now();
    auto duration_dielectric_matrix_mesh = duration_cast<milliseconds>(stop_dielectric_matrix_mesh - start);
    auto start_dielectric_matrix_inversion = high_resolution_clock::now();

    std::cout << "Done in " << duration_dielectric_matrix_mesh.count()/1000.0 << " s.\nInverting the dielectric matrix... " << std::flush;

    #pragma omp parallel for
    for (int iq = 0; iq < Nktotal; iq++){

        this->Invepsilonmatrix_.slice(iq) = arma::solve(this->epsilonmatrix_.slice(iq),auxvec);

        //std::cout << iq << ",";
    }

    // auxvecsol = arma::solve(this->epsilonmatrix_.slice(iq),auxvec);

    //this->Invepsilonmatrix_.slice(iq) = arma::inv(this->epsilonmatrix_.slice(iq));

    //#pragma omp parallel for
    // for (int iq = 0; iq < nq; iq++){
    //     //std::cout << "iq = " << iq << "\n"; 

    //     this->Invepsilonmatrix_.slice(iq) = arma::inv(this->epsilonmatrix_.slice(iq));
    // }


    // std::cout << "\nChi_ (" << this->Gs_(0) << "," << this->Gs_(1) << ")" << "(" << (iq) << ") = " << this->Chimatrix_.slice(iq).row(this->Gs_(0))(this->Gs_(1)) << std::endl;
    // std::cout << "\nepsilon_(" << this->Gs_(0) << "," << this->Gs_(1) << ")" << "(" << (iq) << ") = " << this->epsilonmatrix_.slice(iq).row(this->Gs_(0))(this->Gs_(1)) << std::endl;
    // std::cout << "\nepsilon^(-1)_00 (iq) = " << auxvecsol.row(0)(0) << std::endl;
    // std::cout << "\nepsilon^(-1)_(" << this->Gs_(0) << "," << this->Gs_(1) << ")" << "(" << (iq) << ") = " << auxvecsol.row(this->Gs_(0))(this->Gs_(1)) << std::endl;
    // std::cout << "\nepsilon^(-1)_(" << 3 << "," << 2 << ")" << "(" << (iq) << ") = " << auxvecsol.row(3)(2) << std::endl;

    // int maxg = 9;
    // int precision = 7;
    // for (int g1 = 0; g1 < maxg; ++g1){
    //     for (int g2 = 0; g2 < maxg; ++g2){
    //         const char* Resign = real(auxvecsol.row(g1)(g2)) >= 0 ? "+":"-";
    //         const char* Imsign = imag(auxvecsol.row(g1)(g2)) >= 0 ? "+":"-";
    //         dielectricfile << std::fixed << std::setprecision(precision) << Resign << abs(real(auxvecsol.row(g1)(g2))) << " ";
    //         dielectricfile << std::fixed << std::setprecision(precision) << Imsign << abs(imag(auxvecsol.row(g1)(g2))) << "  ";
    //     }
    //     dielectricfile << "\n";
    // }

    // dielectricfile << "Number of cells each direction: " << this->ncell << "\n";
    // dielectricfile << "Number of reciprocal vectors included: " << nGs << "\n";
    // dielectricfile << "Number of valence/conduction bands included: " << nvalencebands_ << "/" << nconductionbands_ << "\n";
    //dielectricfile << "Dielectric matrix at q = " << system->kpoints.row(iq) << "\n";
    

    auto stop = high_resolution_clock::now();
    auto duration_inversion = duration_cast<milliseconds>(stop - start_dielectric_matrix_inversion);
    auto duration = duration_cast<milliseconds>(stop - start);

    std::cout << "Done in " << duration_inversion.count()/1000.0 << " s." << std::endl;
    std::cout << "Total function run time: " << duration.count()/1000.0 << " s." << std::endl;
    //dielectricfile << "Done in " << duration.count()/1000.0 << " s." << std::endl;
    //dielectricfile.close();
}


/**
 * Method to compute the (G,G') matrix element of the static dielectric function at the specified momentum vector q.
 * @details It creates a file with the name "[systemName].screening" where the dielectric function matrix elements are stored.
 * @param kpointsfile File with the kpoints where we want to obtain the bands. If empty or not specified, then the set of 
 * kpoints coincides with the kmesh
 * @return void
*/
void ExcitonTB::computesingleDielectricFunction() {

    if(mode == "realspace"){
        std::cout << "Real space dielectric function not implemented yet. Exiting." << std::endl;

        std::exit(0);
    }

    std::complex<double> Chi;

    arma::rowvec q = this->q_;

    arma::rowvec g = this->trunreciprocalLattice_.row(this->Gs_(0)); // Sets G
    arma::rowvec g2 = this->trunreciprocalLattice_.row(this->Gs_(1)); // Sets G'

    Chi = computesinglePolarizability(q);

    double potential = coulombFT(q + g);

    double kroneckerdelta = this->Gs_(0) == this->Gs_(1) ? 1 : 0;
        
    std::cout << "Polarizability at q = " << Chi << std::endl;
    std::cout << "Dielectric function at q = " << kroneckerdelta - potential*Chi << std::endl;
}

/**
 * Method to initialize the BSE.
 * @details Calls the more general routine which allows
 * to specify a subset of the complete basis.
 */ 
void ExcitonTB::BShamiltonian(){
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

    uint64_t basisDimBSE = basisStates.n_rows;
    std::cout << "BSE dimension: " << basisDimBSE << std::endl;
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
            arma::rowvec k = system->kpoints.row(k_index);
            arma::rowvec k2 = system->kpoints.row(k2_index);
            D = reciprocalInteractionTerm(coefsK, coefsK2, coefsKQ, coefsK2Q, k, k2, k, k2, this->nReciprocalVectors);
            if(this->exchange){
                X = reciprocalInteractionTerm(coefsK2Q, coefsK2, coefsKQ, coefsK, k2 + Q, k2, k + Q, k, this->nReciprocalVectors);
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
            for(int i = 0; i < system->nk; i++){

                arma::uword vband = bandToIndex.at(valenceBands(v)); // Unsigned integer 
                arma::uword cband = bandToIndex.at(conductionBands(c));

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

    for(int n = 1; n < system->nk/2; n++){
        
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
    for (int i = 0; i < finalBasis.n_rows; i++){
        for (int j = 0; j < initialBasis.n_rows; j++){

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
                D = reciprocalInteractionTerm(coefsK, coefsK2, coefsKQ, coefsK2Q, k, k2, k, k2, this->nReciprocalVectors);
                X = 0;
            }
            
            W(i, j) = - (D - X);                
        };
    };

    double delta = 2.4/(2*ncell); // Adjust delta depending on number of k points
    double rho = targetExciton.pairDensityOfStates(energy, delta);
    cout << "DoS value: " << rho << endl;
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
    for(int i = 0; i < system->nk; i++){
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
    for (int i = 0; i < initialBasis.n_rows; i++){

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
            D = reciprocalInteractionTerm(coefsK, coefsK2, coefsKQ, coefsK2Q, k, k2, k, k2, this->nReciprocalVectors);
            X = 0;
        }
        
        W(i) = - (D - X);                
    };

    double delta = 2.0/targetExciton.system->nk; // Adjust delta depending on number of k points
    double rho = targetExciton.pairDensityOfStates(energy, delta);
    cout << "DoS value: " << rho << endl;
    double hbar = 6.582119624E-16; // Units are eV*s

    transitionRate = (ncell*system->a)*2*PI*std::norm(arma::dot(W, initialState))*rho/hbar;


    return transitionRate;
}

/**
 * Method to print information about the exciton.
 * @return void 
 */
void ExcitonTB::printInformation(){
    cout << std::left << std::setw(30) << "Number of cells: " << ncell << endl;
    cout << std::left << std::setw(30) << "Valence bands:";
    for (int i = 0; i < valenceBands.n_elem; i++){
        cout << valenceBands(i) << " ";
    }
    cout << endl;

    cout << std::left << std::setw(30) << "Conduction bands: ";
    for (int i = 0; i < conductionBands.n_elem; i++){
        cout << conductionBands(i) << " ";
    }
    cout << "\n" << endl;

    cout << std::left << std::setw(30) << "Gauge used: " << gauge << endl;
    cout << std::left << std::setw(30) << "Calculation mode: " << mode << endl;
    if(mode == "reciprocalspace"){
        cout << std::left << std::setw(30) << "nG: " << nReciprocalVectors << endl;
    }
    cout << std::left << std::setw(30) << "Potential: " << potential_ << endl;
    if(exchange){
        cout << std::left << std::setw(30) << "Exchange: " << (exchange ? "True" : "False") << endl;
        cout << std::left << std::setw(30) << "Exchange potential: " << exchangePotential_ << endl;
    }
    if(arma::norm(Q) > 1E-7){
        cout << std::left << std::setw(30) << "Q: "; 
        for (auto qi : Q){
            cout << qi << "  ";
        }
        cout << endl;
    }
    cout << std::left << std::setw(30) << "Scissor cut: " << scissor_ << endl;

    if (this->isscreeningset == true){
        std::cout << std::left << std::setw(40) << "\nNumber of valence bands included: " << this->nvalencebands_ << std::endl;
        std::cout << std::left << std::setw(40) << "Number of conduction bands included: " << this->nconductionbands_ << std::endl;
        if (this->mode == "reciprocalspace"){
            std::cout << std::left << std::setw(40) << "Number of reciprocal lattice vectors: " << this->trunreciprocalLattice_.n_rows << std::endl;
        }
        std::cout << std::left << std::setw(40) << "Gcutoff: " << this->Gcutoff_ << std::endl;
        if (this->function_ == "dielectric" || this->function_ == "polarizability"){
            arma::rowvec G(this->trunreciprocalLattice_ .row(this->Gs_(0)));
            arma::rowvec Gprime(this->trunreciprocalLattice_ .row(this->Gs_(1)));
            std::cout << std::left << std::setw(0) << "Selected G(" << std::setw(0) << this->Gs_(0) << "): " << G(0) << " " << G(1) << " " << G(2) << std::endl;
            std::cout << std::left << std::setw(0) << "Selected G'(" << std::setw(0) << this->Gs_(1) << "): " << Gprime(0) << " " << Gprime(1) << " " << Gprime(2) << std::endl;
        } else if (this->function_ == "none") {
            std::cout << std::left << std::setw(40) << "Number of reciprocal vectors: " << this->nGs << std::endl;
        }
    }
}

/* Method to print information of the inverse of the dielectric matrix in a file.
 * @return void 
 */
void ExcitonTB::writeInverseDielectricMatrix(FILE* textfile){

    int ngs = this->trunreciprocalLattice_.n_rows;
    int nqs = system->nk;

    for(unsigned int i = 0; i < nqs; i++){
        for (unsigned int g = 0; g < ngs; g++){
            for (unsigned int g2 = 0; g2 < ngs; g2++){
                fprintf(textfile, "%11.7lf%11.7lf", real(this->Invepsilonmatrix_.slice(i).row(g)(g2)), imag(this->Invepsilonmatrix_.slice(i).row(g)(g2)));
            }
            fprintf(textfile, "\n");
        }
    }

    // Prints the k points to plot the matrix elements in a grid
    std::string filename_k_points = "kgrid" + std::to_string(this->ncell) + ".dat";

    std::ofstream k_points_file; 

    k_points_file.open(filename_k_points);

    if (!k_points_file.is_open()) { // check if the file was opened successfully
        std::cerr << "Error opening file\n";
        std::cerr << errno << "\n";
    }


    for(unsigned int i = 0; i < nqs; i++){
        auto k = system_->kpoints.row(i);
        k_points_file << k(0) << " " << k(1) << " " << k(2) << std::endl;
    }

    k_points_file.close();
}
}