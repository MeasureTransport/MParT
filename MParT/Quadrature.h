#ifndef MPART_QUADRATURE_H
#define MPART_QUADRATURE_H

#include <math.h>
#include <sstream>
#include <vector>

#include <Kokkos_Core.hpp>

namespace mpart{

namespace QuadError{

    /**
        @brief Flags for controlling how the error between levels is computed for multivariate integrands.
     */
    enum Type {
        First,  ///< Only the first component of the integral estimates will be used
        NormInf,///< The infinity norm (i.e., max of abs) of the difference will be used 
        Norm2,  ///< The L2 norm of the difference will be used
        Norm1   ///< The L1 norm (sum of bas) of the difference will be used
    };
}


template<typename MemorySpace>
class QuadratureBase {

public:

    /** Constructs a quadrature rule with internally managed workspace. */
    KOKKOS_FUNCTION QuadratureBase(unsigned int maxDim, unsigned int workspaceSize) : fdim_(maxDim), 
                                                                                      maxDim_(maxDim), 
                                                                                      workspaceSize_(workspaceSize), 
                                                                                      internalWork_("Workspace", workspaceSize), 
                                                                                      workspace_(internalWork_.data())
    {}

    /** Constructs a quadrature rule with externally managed workspace. */
    KOKKOS_FUNCTION QuadratureBase(unsigned int maxDim, 
                                   unsigned int workspaceSize,
                                   double*      workspace) : fdim_(maxDim),
                                                             maxDim_(maxDim),
                                                             workspaceSize_(workspaceSize),
                                                             workspace_(workspace)
    {}

    KOKKOS_INLINE_FUNCTION void SetWorkspace(double* workspace){workspace_ = workspace;};
    
    KOKKOS_INLINE_FUNCTION unsigned int WorkspaceSize() const{return workspaceSize_;};

    KOKKOS_INLINE_FUNCTION void SetDim(unsigned int fdim){assert(fdim<=maxDim_); fdim_ = fdim;}


protected:

    unsigned int fdim_; // The current length of f(x)
    const unsigned int maxDim_; // The maximum length of f(x) allowed by the cache size
    const unsigned int workspaceSize_; // The total number of doubles that could be used by this quadrature method to store function evaluations during a call to "integrate"

    Kokkos::View<double*,MemorySpace> internalWork_;
    double* workspace_;
};


/**

USAGE
@code
ClenshawCurtisQuadrature quad(5);
integral = quad.Integrate(f, 0,1);
@endcode

@code
Eigen::VectorXd wts, pts;
std::tie(wts, pts) = ClenshawCurtisQuadrature::GetRule(order);
@endcode

 */
template<typename MemorySpace=Kokkos::HostSpace>
class ClenshawCurtisQuadrature : public QuadratureBase<MemorySpace>{
public: 

    /**
     * @brief Construct a new Clenshaw Curtis Quadrature object using an internally allocated workspace.
     * 
     * @param numPts The number of points in the CC rule.
     * @param maxDim The maximum dimension of the integrand.  Used to help set up workspace.
     * @param workspace A pointer to memory that is allocated as a workspace.  Must have space for at least maxDim components.
     */
    KOKKOS_FUNCTION ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim) : QuadratureBase<MemorySpace>(maxDim,maxDim),  pts_("Points", numPts), wts_("Weights", numPts), numPts_(numPts)
    {
        // TODO: Add parallel for loop here with one thread to make sure rule is filled in the correct space
        GetRule(numPts, wts_.data(), pts_.data());
    };

    /**
     * @brief Construct a new Clenshaw Curtis Quadrature object without allocating workspace memory.
     * 
     * @param numPts The number of points in the CC rule.
     * @param maxDim The maximum dimension of the integrand.  Used to help set up workspace.
     * @param workspace A pointer to memory that is allocated as a workspace.  Must have space for at least maxDim components.  Set to null ptr if workspace memory will be allocated later using the SetWorkspace function.
     */
    KOKKOS_FUNCTION ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim, double* workspace) : QuadratureBase<MemorySpace>(maxDim,maxDim,workspace),  pts_("Points", numPts), wts_("Weights", numPts), numPts_(numPts)
    {
        // TODO: Add parallel for loop here with one thread to make sure rule is filled in the correct space
        GetRule(numPts, wts_.data(), pts_.data());
    };

    /**
     @brief Computes the weights and points in a Clenshaw-Curtis rule.
     @param[in] numPts The number of points in the quadrature rule.
     @param[out] wts A pointer to the memory where the weights will be stored.  Must be at least numPts long.
     @param[out] pts A pointer to the memory where the points will be stored.  Must be at least numPts long.
     */
    KOKKOS_FUNCTION static void GetRule(unsigned int numPts, double* wts, double* pts)
    {   
        if(numPts==0){
            return;
        }

        // return results for order == 1
        if(numPts == 1) {
            pts[0] = 0.0;
            wts[0] = 2.0;
            return;
        }

        // define quadrature points
        for (unsigned int i=0; i<numPts; ++i) {
            pts[i] = std::cos( ( numPts - (i+1) ) * M_PI / ( numPts - 1 ) );
        }
        pts[0] = -1.0;
        if ( numPts % 2 == 1) {
            pts[(numPts-1)/2] = 0.0;
        }
        pts[numPts-1] = 1.0;

        // compute quadrature weights 
        double theta;        
        for (unsigned int i=0; i<numPts; ++i) {  
            wts[i] = 1.0;

            theta = i * M_PI / ( numPts - 1 );
            for (unsigned int j=0; j<(numPts-1)/2; ++j){
                double b;
                if ( 2 * (j+1) == ( numPts - 1 ) ) {
                    b = 1.0;
                } else {
                    b = 2.0;
                }
                wts[i] = wts[i] - b * std::cos ( 2.0 * (j+1) * theta ) / ( 4 * (j+1) * (j+1) - 1 );
            }
        }

        // adjust boundary weights
        wts[0] = wts[0] / ( numPts - 1.0 );
        for (unsigned int i=1; i<(numPts-1); ++i){
            wts[i] = 2.0 * wts[i] / ( numPts - 1.0 );
        }
        wts[numPts-1] = wts[numPts-1] / ( numPts - 1.0 );
    }
    
    /**
     @brief Approximates the integral \f$\int_{x_L}^{x_U} f(x) dx\f$ using a Clenshaw-Curtis quadrature rule.
     @details
     @param[in] f The integrand.  Can return any type that overloads multiplication with a double and the += operator.  doubles and Eigen::VectorXd are examples.
     @param[in] lb The lower bound \f$x_L\f$ in the integration.
     @param[in] lb The upper bound \f$x_U\f$ in the integration.
     @param[out] res A pointer to the array where the approximation of \f$\int_{x_L}^{x_U} f(x) dx\f$ should be stored.  Must have at least enough space to store a vector computed by the dimension of f.  Set the dimension of f in the constructor or using the QuadratureBase::SetDim function.
     @tparam FunctionType The type of the integrand.  Must have an operator()(double x) function.
     */
    template<class FunctionType>
    KOKKOS_FUNCTION void Integrate(FunctionType const& f, 
                                   double              lb, 
                                   double              ub,
                                   double*             res)
    {   
        assert(this->workspace_);

        if((ub-lb)<15.0*std::numeric_limits<double>::epsilon()){

            f(0.5*(ub+lb), res);
            for(unsigned int i=0; i<this->fdim_; ++i)
                res[i] *= (ub-lb);
                
            return;
        }

        // Create an output variable
        double* fval = this->workspace_;
        
        // Evaluate integral and store results in output
        for (unsigned int i=0; i<pts_.size(); ++i){
            f(0.5*(ub+lb + (ub-lb)*pts_(i)), fval);
            for(unsigned int j=0; j<this->fdim_; ++j)
                res[j] += 0.5*(ub-lb)*wts_(i) * fval[j];
        }
    }

private:
    const unsigned int numPts_;

    Kokkos::View<double*, MemorySpace> pts_, wts_;

}; // class ClenshawCurtisQuadrature


template<typename MemorySpace>
class RecursiveQuadratureBase : public QuadratureBase<MemorySpace>
{

public:
    
    KOKKOS_FUNCTION RecursiveQuadratureBase(unsigned int maxSub, 
                                            unsigned int maxDim,
                                            unsigned int workspaceSize, 
                                            double absTol, 
                                            double relTol,
                                            QuadError::Type errorMetric) :  QuadratureBase<MemorySpace>(maxDim, workspaceSize),
                                                                            maxSub_(maxSub), 
                                                                            absTol_(absTol),
                                                                            relTol_(relTol),
                                                                            errorMetric_(errorMetric)
    {}

    KOKKOS_FUNCTION RecursiveQuadratureBase(unsigned int maxSub, 
                                            unsigned int maxDim,
                                            unsigned int workspaceSize,
                                            double*      workspace, 
                                            double       absTol, 
                                            double       relTol,
                                            QuadError::Type errorMetric) :  QuadratureBase<MemorySpace>(maxDim, workspaceSize, workspace),
                                                                            maxSub_(maxSub), 
                                                                            absTol_(absTol),
                                                                            relTol_(relTol),
                                                                            errorMetric_(errorMetric)
    {}


protected:

    KOKKOS_FUNCTION void EstimateError(const double* coarseVal,
                                       const double* fineVal,
                                       double      & error,
                                       double      & tol) const
    {   
        double relRefVal;
        if(errorMetric_==QuadError::First){   
            error = fabs(fineVal[0]-coarseVal[0]);
            relRefVal = fabs(coarseVal[0]);
        }else if(errorMetric_==QuadError::NormInf){
            error = 0;
            relRefVal = 0;
            for(unsigned int i=0; i<this->fdim_; ++i){
                error = fmax(error, fabs(fineVal[i]-coarseVal[i]));
                relRefVal = fmax(relRefVal, fabs(coarseVal[i]));
            }

        }else if(errorMetric_==QuadError::Norm2){
            
            error = 0;
            relRefVal = 0;
            for(unsigned int i=0; i<this->fdim_; ++i){
                error += (fineVal[i]-coarseVal[i])*(fineVal[i]-coarseVal[i]);
                relRefVal += coarseVal[i]*coarseVal[i];
            }
            error = sqrt(error);
            relRefVal = sqrt(relRefVal);

        }else if(errorMetric_==QuadError::Norm1){

            error = 0;
            relRefVal = 0;
            for(unsigned int i=0; i<this->fdim_; ++i){
                error += fabs(fineVal[i]-coarseVal[i]);
                relRefVal += fabs(coarseVal[i]);
            }

        }
        tol = std::fmax( relTol_*relRefVal, absTol_);
    }

    const unsigned int maxSub_;
    const double absTol_;
    const double relTol_;

    QuadError::Type errorMetric_;
};



/**
 @brief Adaptive Simpson-rule integration based on applying a simple Simpson 1/3 rule recursively on subintervals.

 */
template<typename MemorySpace=Kokkos::HostSpace>
class AdaptiveSimpson : public RecursiveQuadratureBase<MemorySpace> {
public:

    KOKKOS_FUNCTION AdaptiveSimpson(unsigned int maxSub,
                                    unsigned int fdim, 
                                    double absTol, 
                                    double relTol,
                                    QuadError::Type errorMetric) : RecursiveQuadratureBase<MemorySpace>(maxSub, fdim, 4*(maxSub+1)*fdim, absTol, relTol, errorMetric){};

    KOKKOS_FUNCTION AdaptiveSimpson(unsigned int maxSub,
                                    unsigned int fdim, 
                                    double* workspace,
                                    double absTol, 
                                    double relTol,
                                    QuadError::Type errorMetric) : RecursiveQuadratureBase<MemorySpace>(maxSub, fdim, 4*(maxSub+1)*fdim, workspace, absTol, relTol, errorMetric){};


    /**
     @brief Approximates the integral \f$\int_{x_L}^{x_U} f(x) dx\f$
     @details
     @param[in] f The integrand.  Can return any type that overloads multiplication with a double and the += operator.  doubles and Eigen::VectorXd are examples.
     @param[in] lb The lower bound \f$x_L\f$ in the integration.
     @param[in] lb The upper bound \f$x_U\f$ in the integration.
     @returns An approximation of \f$\int_{x_L}^{x_U} f(x) dx\f$.  The return type will be the same as the type returned by the integrand function.
     @tparam FunctionType The type of the integrand.  Must have an operator()(double x) function.
     */
    template<class FunctionType>
    KOKKOS_FUNCTION void Integrate(FunctionType const& f, 
                                   double              lb, 
                                   double              ub,
                                   double*             res)
    {   
        assert(this->workspace_);

        auto flb = &this->workspace_[0];
        auto fub = &this->workspace_[this->fdim_];
        auto fmb = &this->workspace_[2*this->fdim_];
        auto intCoarse = &this->workspace_[3*this->fdim_];

        f(lb, flb);
        f(ub, fub);

        double midPt = 0.5*(lb+ub);
        f(midPt, fmb);
        
        // Compute ((ub-lb)/6.0) * (flb + 4.0*fmb + fub);
        for(unsigned int i=0; i<this->fdim_; ++i) 
            intCoarse[i] = ((ub-lb)/6.0) * (flb[i] + 4.0*fmb[i] + fub[i]);

        int status = 1; 
        unsigned int maxLevel = 0;
        RecursiveIntegrate(f, lb, midPt, ub, flb, fmb, fub, 0, intCoarse, res, status, maxLevel);
    }

private:

    template<class ScalarFuncType>
    KOKKOS_FUNCTION void RecursiveIntegrate(ScalarFuncType const& f, 
                            double          leftPt,
                            double          midPt,
                            double          rightPt,
                            const double*   leftFunc, 
                            const double*   midFunc,
                            const double*   rightFunc,
                            int             level,
                            const double*   intCoarse,
                            double*         integral, 
                            int           & status, 
                            unsigned int  & levelOut)
    {
        if((rightPt-leftPt)<15.0*std::numeric_limits<double>::epsilon()){

            for(unsigned int i=0; i<this->fdim_; ++i)
                integral[i] = (rightPt-leftPt)*midFunc[i];
            status = -1;
            levelOut = level;
            return;
        }

        // update current refinement level
        level += 1;
        
        // evaluate integral on each sub-interval
        double* leftMidFunc = &this->workspace_[(4*level) * this->fdim_];
        double* rightMidFunc = &this->workspace_[(4*level+1) * this->fdim_];
        double* intFinerLeft = &this->workspace_[(4*level+2) * this->fdim_];
        double* intFinerRight = &this->workspace_[(4*level+3) * this->fdim_];

        double leftMidPt = 0.5*(leftPt+midPt);
        f(leftMidPt, leftMidFunc);

        double rightMidPt = 0.5*(midPt+rightPt);
        f(rightMidPt, rightMidFunc);

        for(unsigned int i=0; i<this->fdim_; ++i){
            intFinerLeft[i]  = ((midPt-leftPt)/6.0) * (leftFunc[i] + 4.0*leftMidFunc[i] + midFunc[i]);
            intFinerRight[i] = ((rightPt-midPt)/6.0) * (midFunc[i] + 4.0*rightMidFunc[i] + rightFunc[i]);
            integral[i] = intFinerLeft[i] + intFinerRight[i];
        }

        // Compute error and tolerance
        double intErr, tol;
        this->EstimateError(intCoarse, integral, intErr, tol);

        // Stop the recursion if the level hit maximum depth
        if ( (intErr > tol) && (level == this->maxSub_) ) {
            status = -1;
            levelOut = level;
            return;
        }
        // If the error between levels is smaller than Tol, return the finer level result
        else if ( intErr <= tol ) {
            status = 1;
            levelOut = level;
            return;
        }
        // Else subdivide further
        else {
            //OutputType intLeft, intRight;
            int statusLeft, statusRight;
            unsigned int levelLeft, levelRight;

            RecursiveIntegrate(f, leftPt, leftMidPt, midPt, leftFunc, leftMidFunc, midFunc, level, intFinerLeft, integral, statusLeft, levelLeft);

            // At this point, we no longer need intFinerLeft, so use that memory to store the new fine value of the right integral
            RecursiveIntegrate(f, midPt, rightMidPt, rightPt, midFunc, rightMidFunc, rightFunc, level, intFinerRight, intFinerLeft, statusRight, levelRight);

            // Add the left and right integrals
            for(unsigned int i=0; i<this->fdim_; ++i)
                integral[i] += intFinerLeft[i];

            status = int( ((statusLeft<0)||(statusLeft<0))?-1:1 );
            levelOut =  fmax(levelLeft, levelRight);
            return; 
        }
    }

}; // Class AdaptiveSimpsonMixin


/**
 @brief Adaptive quadrature based on applying a Clenshaw-Curtis recursively on subintervals.
 @details The points in a Clenshaw-Curtis quadrature rule with \f$2^{L}+1\f$ points are a subset of the points in a rule with \f$2^{L+1}+1\f$ points.
          This "nesting" allows us to approximate the integral at two different levels with minimal additional integrand evaluations.  Comparing the
          integral at these values gives an indication of the error in the integral approximation.  If the error is unacceptable large, we can subdivide
          the integration domain and apply Clenshaw-Curtis rules on each subinterval.   This class implements a recursive version of this process.  Two nested
          Clenshaw-Curtis rules are used to estimate the integral, and its error.  If the error is too large, then the integration domain is split into 
          two equal halves, where the nested quadrature rules can again be applied.  This recursive subdivision repeats until an acceptable error level
          is reached or until a maximum number of subdivisions has occured.
 */
template<typename MemorySpace=Kokkos::HostSpace>
class AdaptiveClenshawCurtis : public RecursiveQuadratureBase<MemorySpace>{
public:

    /**
       @brief Construct a new adaptive quadrature class with specified stopping criteria.
       @param maxSub The maximum number of subintervals allowed.
       @param maxDim The maximum dimension of the integrand.
       @param[in] absTol An absolute error tolerance used to stop the adaptive integration.
       @param[in] relTol A relative error tolerance used to stop te adaptive integration.
       @param[in] errorMetric A flag specifying the type of error metric to use.
       @param[in] level The nesting level \f$L\f$.  A coarse rule with \f$2^{L}+1\f$ points will be used and a fine rule with \f$w^{L+1}+1\f$ points will be used.
     */
   KOKKOS_FUNCTION AdaptiveClenshawCurtis(unsigned int      level,
                           unsigned int      maxSub,
                           unsigned int      maxDim, 
                           double            absTol, 
                           double            relTol,
                           QuadError::Type   errorMetric) : RecursiveQuadratureBase<MemorySpace>(maxSub, maxDim, 3*(maxSub+1)*maxDim , absTol, relTol, errorMetric),
                                                            coarsePts_("Coarse Pts", std::pow(2,level)+1),
                                                            coarseWts_("Coarse Wts", std::pow(2,level)+1),
                                                            finePts_("Fine Pts", std::pow(2,level+1)+1),
                                                            fineWts_("Coarse Pts", std::pow(2,level+1)+1)
    {
        assert(std::pow(2,level)+1 >=3);
        ClenshawCurtisQuadrature<MemorySpace>::GetRule(std::pow(2,level)+1,  coarseWts_.data(), coarsePts_.data());
        ClenshawCurtisQuadrature<MemorySpace>::GetRule(std::pow(2,level+1)+1,  fineWts_.data(), finePts_.data());
    };

    KOKKOS_FUNCTION AdaptiveClenshawCurtis(unsigned int      level,
                           unsigned int      maxSub,
                           unsigned int      maxDim, 
                           double*           workspace,
                           double            absTol, 
                           double            relTol,
                           QuadError::Type   errorMetric) : RecursiveQuadratureBase<MemorySpace>(maxSub, maxDim, 3*(maxSub+1)*maxDim, workspace, absTol, relTol, errorMetric),
                                                            coarsePts_("Coarse Pts", std::pow(2,level)+1),
                                                            coarseWts_("Coarse Wts", std::pow(2,level)+1),
                                                            finePts_("Fine Pts", std::pow(2,level+1)+1),
                                                            fineWts_("Coarse Pts", std::pow(2,level+1)+1)
    {
        assert(std::pow(2,level)+1 >=3);
        ClenshawCurtisQuadrature<MemorySpace>::GetRule(std::pow(2,level)+1,  coarseWts_.data(), coarsePts_.data());
        ClenshawCurtisQuadrature<MemorySpace>::GetRule(std::pow(2,level+1)+1,  fineWts_.data(), finePts_.data());
    };



    /**
     @brief Approximates the integral \f$\int_{x_L}^{x_U} f(x) dx\f$
     @details
     @param[in] f The integrand.  Can return any type that overloads multiplication with a double and the += operator.  doubles and Eigen::VectorXd are examples.
     @param[in] lb The lower bound \f$x_L\f$ in the integration.
     @param[in] lb The upper bound \f$x_U\f$ in the integration.
     @returns An approximation of \f$\int_{x_L}^{x_U} f(x) dx\f$.  The return type will be the same as the type returned by the integrand function.
     @tparam FunctionType The type of the integrand.  Must have an operator()(double x) function.
     */
    template<class FunctionType>
    KOKKOS_FUNCTION void Integrate(FunctionType const& f, 
                                   double              lb, 
                                   double              ub,
                                   double*             res)
    {   
        assert(this->workspace_);

        int status;
        unsigned int maxLevel;

        double* leftVal = &this->workspace_[0];
        f(lb,leftVal);

        double* rightVal = &this->workspace_[this->fdim_];
        f(ub, rightVal);

        RecursiveIntegrate(f, lb, ub, leftVal, rightVal, 0, res, status, maxLevel);
    }

private:

    template<class ScalarFuncType>
    KOKKOS_FUNCTION void RecursiveIntegrate(ScalarFuncType const& f, 
                                            double                leftPt,
                                            double                rightPt,
                                            const double*         leftFunc, 
                                            const double*         rightFunc,
                                            int                   level,
                                            double*               integral, 
                                            int                 & status, 
                                            unsigned int        & levelOut)
    {
        // update current refinement level
        level += 1;
        
        // Figure out where we are in the workspace
        double* fval = &this->workspace_[(3*level)*this->fdim_];
        double* midFunc = &this->workspace_[(3*level +1)*this->fdim_];
        double* intCoarse = &this->workspace_[(3*level + 2)*this->fdim_];

        const double midPt = 0.5*(rightPt+leftPt);
        const double scale = 0.5*(rightPt-leftPt);

        // Evaluate the midpoint
        f(midPt, midFunc);

        if(scale<15.0*std::numeric_limits<double>::epsilon()){

            for(unsigned int i=0; i<this->fdim_; ++i)
                integral[i] = (rightPt-leftPt)*midFunc[i];

            status = -1;
            levelOut = level;
            return;
        }

        // Start with the left, right, and middle points
        unsigned int coarseRightIndex = this->coarseWts_.extent(0)-1;
        unsigned int coarseMidIndex = coarseRightIndex/2;
        unsigned int fineRightIndex = this->fineWts_.extent(0)-1;
        unsigned int fineMidIndex = fineRightIndex/2;

        for(unsigned int i=0; i<this->fdim_; ++i){
            intCoarse[i] = scale*this->coarseWts_(0) * leftFunc[i];
            intCoarse[i] += scale*this->coarseWts_(coarseMidIndex) * midFunc[i];
            intCoarse[i] += scale*this->coarseWts_(coarseRightIndex) * rightFunc[i];

            integral[i] =scale*this->fineWts_(0) * leftFunc[i];
            integral[i] += scale*this->fineWts_(fineMidIndex) * midFunc[i];
            integral[i] += scale*this->fineWts_(fineRightIndex) * rightFunc[i];
        }

        // Compute the coarse integral, and the part of the fine integral coming from the nested coarse points 
        for(unsigned int i=1; i<coarseMidIndex; ++i){
            f(midPt + scale*this->coarsePts_(i), fval);

            for(unsigned int j=0; j<this->fdim_; ++j){
                intCoarse[j] += scale*this->coarseWts_(i) * fval[j];
                integral[j] += scale*this->fineWts_(2*i) * fval[j];
            }
        }

        for(unsigned int i=coarseMidIndex+1; i<coarseRightIndex; ++i){
            f(midPt + scale*this->coarsePts_(i), fval);

            for(unsigned int j=0; j<this->fdim_; ++j){
                intCoarse[j] += scale*this->coarseWts_(i) * fval[j];
                integral[j] += scale*this->fineWts_(2*i) * fval[j];
            }
        }

        // Now add all the points that are only in the fine rule
        for (unsigned int i=1; i<this->finePts_.extent(0); i+=2){
            f(midPt + scale*this->finePts_(i), fval);
            for(unsigned int j=0; j<this->fdim_; ++j)
                integral[j] += scale*this->fineWts_(i) * fval[j];
        }

        // Compute error and tolerance
        double intErr, tol;
        this->EstimateError(intCoarse, integral, intErr, tol);

        // Stop the recursion if the level hit maximum depth
        if ( (intErr > tol) && (level == this->maxSub_) ) {
            status = -1;
            levelOut = level;
            return;
        }
        // If the error between levels is smaller than Tol, return the finer level result
        else if ( intErr <= tol ) {
            status = 1;
            levelOut = level;
            return;
        }
        // Else subdivide further
        else {
            //OutputType intLeft, intRight;
            int statusLeft, statusRight;
            unsigned int levelLeft, levelRight;
            double midPt = 0.5*(leftPt+rightPt);

            RecursiveIntegrate(f, leftPt, midPt, leftFunc, midFunc, level, integral, statusLeft, levelLeft);
           
            // At this point, we no longer need fval, so use that memory to store the new fine value of the right integral
            RecursiveIntegrate(f, midPt, rightPt, midFunc, rightFunc, level, fval, statusRight, levelRight);

            // Add the left and right integrals
            for(unsigned int i=0; i<this->fdim_; ++i)
                integral[i] += fval[i];

            status = int( ((statusLeft<0)||(statusLeft<0))?-1:1 );
            levelOut =  fmax(levelLeft, levelRight);
            return; 
        }

    }


    Kokkos::View<double*,MemorySpace> coarsePts_, coarseWts_, finePts_, fineWts_;

}; // class AdaptiveClenshawCurtis


} // namespace mpart

#endif 