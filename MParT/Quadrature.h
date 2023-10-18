#ifndef MPART_QUADRATURE_H
#define MPART_QUADRATURE_H

#include <math.h>
#include <sstream>
#include <vector>

#include <Kokkos_Core.hpp>

#include <Eigen/Core>

#if defined(MPART_HAS_CEREAL)
#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#endif 

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
#if defined(MPART_HAS_CEREAL)
    friend class cereal::access;
#endif 

    /** Constructs a quadrature rule with internally managed workspace. */
    QuadratureBase(unsigned int maxDim, unsigned int workspaceSize) : fdim_(maxDim),
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

#if defined(MPART_HAS_CEREAL)
    // Define a serialize or save/load pair as you normally would
    template <class Archive>
    void save( Archive & ar ) const
    {   
        ar(fdim_, maxDim_, workspaceSize_);
        ar(internalWork_);
    }

    template <class Archive>
    void load( Archive & ar )
    {   
        ar(fdim_, maxDim_, workspaceSize_);
        ar(internalWork_);

        // Set the workspace point if applicable
        if(internalWork_.is_allocated()){
            if(internalWork_.size()>=workspaceSize_){
                workspace_ = internalWork_.data();
            }
        }
    }
    QuadratureBase() : workspace_(nullptr){};
#endif    

protected:

    unsigned int fdim_; // The current length of f(x)
    unsigned int maxDim_; // The maximum length of f(x) allowed by the cache size
    unsigned int workspaceSize_; // The total number of doubles that could be used by this quadrature method to store function evaluations during a call to "integrate"

    Kokkos::View<double*,MemorySpace> internalWork_;
    double* workspace_;
};


/**

USAGE

With internally constructed workspace:
@code
unsigned int fdim = 1; // Size of vector-valued integrand.  Set to 1 for scalar integrand
auto f = [](double x, double* fval){ fval[0] = x*x;};

ClenshawCurtisQuadrature quad(5,fdim);

double integral;
quad.Integrate(f, 0, 1, &integral); // Integrate x^2 from 0 to 1 and store in "integral"
@endcode

With external workspace specified at construction
@code
unsigned int fdim = 1;
auto f = [](double x, double* fval){ fval[0] = x*x;};

std::vector<double> workspace(ClenshawCurtisQuadrature::GetWorkspaceSize(fdim));

ClenshawCurtisQuadrature quad(5, fdim, *workspace[0]);

double integral;
quad.Integrate(f, 0, 1, &integral); // Integrate x^2 from 0 to 1 and store in "integral"
@endcode

With external workspace specified at integration time.  This is often useful when integration is performed in a Kokkos parallel_for loop.
@code
unsigned int fdim = 1;
auto f = [](double x, double* fval){ fval[0] = x*x;};

ClenshawCurtisQuadrature quad(5, fdim, nullptr);

std::vector<double> workspace(ClenshawCurtisQuadrature::GetWorkspaceSize(fdim));

double integral;
quad.Integrate(&workspace[0], f, 0, 1, &integral); // Integrate x^2 from 0 to 1 and store in "integral"
@endcode
 */
template<typename MemorySpace=Kokkos::HostSpace>
class ClenshawCurtisQuadrature : public QuadratureBase<MemorySpace>{
public:
#if defined(MPART_HAS_CEREAL)
    friend class cereal::access;
#endif 

    /**
     * @brief Construct a new Clenshaw Curtis Quadrature object using an internally allocated workspace.
     *
     * @param numPts The number of points in the CC rule.
     * @param maxDim The maximum dimension of the integrand.  Used to help set up workspace.
     * @param workspace A pointer to memory that is allocated as a workspace.  Must have space for at least maxDim components.
     */
    inline ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim);

    /**
        @brief Construct a new Clenshaw Curtis Quadrature object with externally allocated workspace memory.

        @param numPts The number of points in the CC rule.
        @param maxDim The maximum dimension of the integrand.  Used to help set up workspace.
        @param workspace A pointer to memory that is allocated as a workspace.  Must have space for at least maxDim components.  Set to null ptr if workspace memory will be allocated later using the SetWorkspace function.
    */
    inline ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim, double* workspace);

    /** @brief Returns the size of a double array needed as a workspace to integrate a function with dimension fdim.
        @param fdim The dimension of a vector-valued integrand.
        @return The minimum length of a double array needed for the workspace.
    */
    KOKKOS_FUNCTION static unsigned int GetWorkspaceSize(unsigned int fdim){return fdim;};

    KOKKOS_INLINE_FUNCTION void SetDim(unsigned int fdim){

        // If the workspace is managed internally, we need to make sure we won't required too much memory
        if(this->internalWork_.extent(0)>0)
            assert(fdim<=this->maxDim_);

        this->fdim_ = fdim;
        this->workspaceSize_=GetWorkspaceSize(fdim);
        assert((this->internalWork_.extent(0)==0)||(this->internalWork_.extent(0)>=this->workspaceSize_));
    }

    /**
     * @brief Construct a new Clenshaw Curtis Quadrature object without allocating workspace memory and specifying views where the points and weights should be stored.
     *
       @param pts A view to hold the points in the quadrature rule.  Does not need to contain the points themselves, just memory to store them. The length of this view dictates how many points are in the quadrature rule.
       @param wts A view to hold the weights in the quadrature rule.  Like the pts view, this does not need to contain the weights, just space to store them.  Must be the same length as pts.
       @param numPts The number of points in the CC rule.
       @param maxDim The maximum dimension of the integrand.  Used to help set up workspace.
       @param workspace A pointer to memory that is allocated as a workspace.  Must have space for at least maxDim components.  Set to null ptr if workspace memory will be allocated later using the SetWorkspace function.
     */
    KOKKOS_FUNCTION ClenshawCurtisQuadrature(Kokkos::View<double*, MemorySpace> pts, Kokkos::View<double*, MemorySpace> wts, unsigned int maxDim, double* workspace) : QuadratureBase<MemorySpace>(maxDim,maxDim,workspace),  pts_(pts), wts_(wts), numPts_(pts.extent(0))
    {
        GetRule(numPts_, wts_.data(), pts_.data());
    };

    static std::pair<Eigen::VectorXd, Eigen::VectorXd> GetRule(unsigned int order)
    {
        Eigen::VectorXd wts(order), pts(order);
        ClenshawCurtisQuadrature::GetRule(order, wts.data(), pts.data());
        return std::make_pair(wts,pts);
    }

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
                                   double*             res) const
    {
        assert(this->workspace_);
        Integrate(this->workspace_, f, lb, ub, res);
    }


    template<class FunctionType>
    KOKKOS_FUNCTION void Integrate(double*             workspace,
                                   FunctionType const& f,
                                   double              lb,
                                   double              ub,
                                   double*             res) const
    {

        // For sufficiently small interval, use the midpoint rule
        double midpoint_tol = 15.0*std::numeric_limits<double>::epsilon();
        if((ub-lb)<midpoint_tol){

            f(0.5*(ub+lb), res);
            for(unsigned int i=0; i<this->fdim_; ++i)
                res[i] *= (ub-lb);

            return;
        }

        for(unsigned int j=0; j<this->fdim_; ++j)
            res[j] = 0.0;

        // Create an output variable
        double* fval = workspace;

        // Evaluate integral and store results in output
        for (unsigned int i=0; i<pts_.size(); ++i){
            f(0.5*(ub+lb + (ub-lb)*pts_(i)), fval);
            for(unsigned int j=0; j<this->fdim_; ++j)
                res[j] += 0.5*(ub-lb)*wts_(i) * fval[j];
        }
    }

#if defined(MPART_HAS_CEREAL)
    // Define a serialize or save/load pair as you normally would
    template <class Archive>
    void save( Archive & ar ) const
    {   
        ar(cereal::base_class<QuadratureBase<MemorySpace>>(this));
        ar(pts_,wts_, numPts_);
    }
    template <class Archive>
    void load( Archive & ar )
    {   
        ar(cereal::base_class<QuadratureBase<MemorySpace>>(this));
        ar(pts_,wts_, numPts_);
    }

    /** Default constructor should never be used directly.  Only for serialization. */
    ClenshawCurtisQuadrature(){};

#endif 

private:
    
    Kokkos::View<double*, MemorySpace> pts_, wts_;
    unsigned int numPts_;

}; // class ClenshawCurtisQuadrature


template<typename MemorySpace>
struct GetRuleFunctor{

    GetRuleFunctor(unsigned int numPts, double* pts, double* wts) : numPts_(numPts), pts_(pts), wts_(wts){};

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const{
        ClenshawCurtisQuadrature<MemorySpace>::GetRule(numPts_,wts_, pts_);
    };

    unsigned int numPts_;
    double* pts_;
    double* wts_;
};

template<typename MemorySpace>
inline ClenshawCurtisQuadrature<MemorySpace>::ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim) : QuadratureBase<MemorySpace>(maxDim,maxDim),  pts_("Points", numPts), wts_("Weights", numPts), numPts_(numPts)
{
    // TODO: Add parallel for loop here with one thread to make sure rule is filled in the correct space
    Kokkos::parallel_for(1, GetRuleFunctor<MemorySpace>(numPts, pts_.data(), wts_.data()));
};

template<>
inline ClenshawCurtisQuadrature<Kokkos::HostSpace>::ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim) : QuadratureBase<Kokkos::HostSpace>(maxDim,maxDim),  pts_("Points", numPts), wts_("Weights", numPts), numPts_(numPts)
{
    GetRule(numPts, wts_.data(), pts_.data());
};


template<typename MemorySpace>
inline ClenshawCurtisQuadrature<MemorySpace>::ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim, double* workspace) : QuadratureBase<MemorySpace>(maxDim,maxDim,workspace),  pts_("Points", numPts), wts_("Weights", numPts), numPts_(numPts)
{
    // TODO: Add parallel for loop here with one thread to make sure rule is filled in the correct space
    Kokkos::parallel_for(1, GetRuleFunctor<MemorySpace>(numPts, pts_.data(), wts_.data()));
};

template<>
inline ClenshawCurtisQuadrature<Kokkos::HostSpace>::ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim, double* workspace) : QuadratureBase<Kokkos::HostSpace>(maxDim,maxDim,workspace),  pts_("Points", numPts), wts_("Weights", numPts), numPts_(numPts)
{
    GetRule(numPts, wts_.data(), pts_.data());
};


template<typename MemorySpace>
class RecursiveQuadratureBase : public QuadratureBase<MemorySpace>
{

public:
#if defined(MPART_HAS_CEREAL)
    friend class cereal::access;
#endif 

    RecursiveQuadratureBase(unsigned int maxSub,
                            unsigned int maxDim,
                            unsigned int workspaceSize,
                            double absTol,
                            double relTol,
                            QuadError::Type errorMetric,
                            unsigned int minSub=0) :  QuadratureBase<MemorySpace>(maxDim, workspaceSize),
                                                            maxSub_(maxSub),
                                                            absTol_(absTol),
                                                            relTol_(relTol),
                                                            errorMetric_(errorMetric),
                                                            minSub_(minSub)
    {}

    KOKKOS_FUNCTION RecursiveQuadratureBase(unsigned int maxSub,
                                            unsigned int maxDim,
                                            unsigned int workspaceSize,
                                            double*      workspace,
                                            double       absTol,
                                            double       relTol,
                                            QuadError::Type errorMetric,
                                            unsigned int    minSub=0) :  QuadratureBase<MemorySpace>(maxDim, workspaceSize, workspace),
                                                                            maxSub_(maxSub),
                                                                            absTol_(absTol),
                                                                            relTol_(relTol),
                                                                            errorMetric_(errorMetric),
                                                                            minSub_(minSub)
    {}


#if defined(MPART_HAS_CEREAL)
    // Define a serialize or save/load pair as you normally would
    template <class Archive>
    void save( Archive & ar ) const
    {   
        ar(cereal::base_class<QuadratureBase<MemorySpace>>(this));
        ar(maxSub_, minSub_, absTol_, relTol_, errorMetric_);
    }
    template <class Archive>
    void load( Archive & ar )
    {   
        ar(cereal::base_class<QuadratureBase<MemorySpace>>(this));
        ar(maxSub_, minSub_, absTol_, relTol_, errorMetric_);
    }
#endif 

protected:
    RecursiveQuadratureBase(){};

    KOKKOS_FUNCTION void EstimateError(const double* coarseVal,
                                       const double* fineVal,
                                       double      & error,
                                       double      & tol) const
    {
        double relRefVal;
        switch(errorMetric_){
            case QuadError::First:
                error = fabs(fineVal[0]-coarseVal[0]);
                relRefVal = fabs(coarseVal[0]);
                break;
            case QuadError::NormInf:
                error = 0;
                relRefVal = 0;
                for(unsigned int i=0; i<this->fdim_; ++i){
                    error = fmax(error, fabs(fineVal[i]-coarseVal[i]));
                    relRefVal = fmax(relRefVal, fabs(coarseVal[i]));
                }
                break;
            case QuadError::Norm2:
                error = 0;
                relRefVal = 0;
                for(unsigned int i=0; i<this->fdim_; ++i){
                    error += (fineVal[i]-coarseVal[i])*(fineVal[i]-coarseVal[i]);
                    relRefVal += coarseVal[i]*coarseVal[i];
                }
                error = sqrt(error);
                relRefVal = sqrt(relRefVal);
                break;
            default: // Norm1 is default
                error = 0;
                relRefVal = 0;
                for(unsigned int i=0; i<this->fdim_; ++i){
                    error += fabs(fineVal[i]-coarseVal[i]);
                    relRefVal += fabs(coarseVal[i]);
                }
                break;
        }

        tol = std::fmax( relTol_*relRefVal, absTol_);
    }

    unsigned int maxSub_;
    unsigned int minSub_;
    double absTol_;
    double relTol_;

    QuadError::Type errorMetric_;
};



/**
 @brief Adaptive Simpson-rule integration based on applying a simple Simpson 1/3 rule recursively on subintervals.

 */
template<typename MemorySpace=Kokkos::HostSpace>
class AdaptiveSimpson : public RecursiveQuadratureBase<MemorySpace> {
public:
#if defined(MPART_HAS_CEREAL)
    friend class cereal::access;
#endif 

    AdaptiveSimpson(unsigned int maxSub,
                    unsigned int fdim,
                    double absTol,
                    double relTol,
                    QuadError::Type errorMetric,
                    unsigned int minSub=0) : RecursiveQuadratureBase<MemorySpace>(maxSub, fdim, GetWorkspaceSize(maxSub,fdim), absTol, relTol, errorMetric,minSub){};

    KOKKOS_FUNCTION AdaptiveSimpson(unsigned int maxSub,
                                    unsigned int fdim,
                                    double* workspace,
                                    double absTol,
                                    double relTol,
                                    QuadError::Type errorMetric, 
                                    unsigned int minSub=0) : RecursiveQuadratureBase<MemorySpace>(maxSub, fdim, GetWorkspaceSize(maxSub,fdim), workspace, absTol, relTol, errorMetric, minSub){};


    KOKKOS_FUNCTION static unsigned int GetWorkspaceSize(unsigned int maxSub, unsigned int fdim){return (2*maxSub+5)*fdim + 2*maxSub;};


    KOKKOS_INLINE_FUNCTION void SetDim(unsigned int fdim){

        // If the workspace is managed internally, we need to make sure we won't required too much memory
        if(this->internalWork_.extent(0)>0)
            assert(fdim<=this->maxDim_);

        this->fdim_ = fdim;
        this->workspaceSize_=GetWorkspaceSize(this->maxSub_, fdim);
        assert((this->internalWork_.extent(0)==0)||(this->internalWork_.extent(0)>=this->workspaceSize_));
    }

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
                                   double*             res) const
    {
        assert(this->workspace_);
        Integrate(this->workspace_, f, lb, ub, res);
    }

    template<class FunctionType>
    KOKKOS_FUNCTION void Integrate(double* workspace,
                                   FunctionType const& f,
                                   double              lb,
                                   double              ub,
                                   double*             res) const
    {
        for(unsigned int i=0; i<this->fdim_; ++i){
            res[i] = 0;
        }

        unsigned int currLevel = 0;

        unsigned int currSegment = 0;

        double midPt = 0.5*(lb+ub);

        double* leftFunc = &workspace[0];
        f(lb, leftFunc);

        double* rightFunc = &workspace[this->fdim_];
        f(ub, rightFunc);

        double* midFunc = &workspace[2*this->fdim_];
        f(midPt, midFunc);

        double* intCoarse = &workspace[3*this->fdim_];
        double* intFine = &workspace[4*this->fdim_];

        double* leftPt  = &workspace[5*this->fdim_];
        *leftPt = lb;
        double* rightPt  = &workspace[5*this->fdim_+1];
        *rightPt = ub;

        double leftMidPt;
        double* leftMidFunc;

        double rightMidPt;
        double* rightMidFunc;

        unsigned int workStartInd =  5*this->fdim_;
        unsigned int prevStartInd = 0;

        double error, errorTol;

        while(true){


            leftPt = &workspace[workStartInd];
            rightPt = &workspace[workStartInd+1];


            // Compute the subinterval mid points
            midPt = 0.5*((*leftPt) + (*rightPt));
            leftMidPt = 0.5*((*leftPt)+midPt);
            rightMidPt = 0.5*(midPt+(*rightPt));


            // Evaluate the integrand at the subinterval mid points
            leftMidFunc = &workspace[workStartInd + 2];
            f(leftMidPt, leftMidFunc);

            rightMidFunc = &workspace[workStartInd + 2 + this->fdim_];
            f(rightMidPt, rightMidFunc);

            // Compute the coarse and fine integrals
            for(unsigned int i=0; i<this->fdim_; ++i){
                intCoarse[i] = (((*rightPt)-(*leftPt))/6.0) * (leftFunc[i] + 4.0*midFunc[i] + rightFunc[i]);
                intFine[i]  = ((midPt-(*leftPt))/6.0) * (leftFunc[i] + 4.0*leftMidFunc[i] + midFunc[i]);
                intFine[i] += (((*rightPt)-midPt)/6.0) * (midFunc[i] + 4.0*rightMidFunc[i] + rightFunc[i]);
            }

            // Check to see if the error is small enough or if we've hit the maximum number of subdivisions
            this->EstimateError(intCoarse, intFine, error, errorTol);

            // Checking for convergence or other termination criteria
            if((((error<errorTol)||(currLevel==this->maxSub_-1))&&(currLevel>=this->minSub_))||(std::abs(ub-lb)<1e-14)){

                for(unsigned int i=0; i<this->fdim_; ++i){
                    res[i] += intFine[i];
                }

                // Find the lowest previous level where we are working on the "left" branch of the tree.
                while(((currSegment >> currLevel) & 1U)&&(currLevel>0)){
                    currSegment &= ~(1UL << currLevel); // Set the side on this level to the left
                    currLevel--;
                }

                // If we're back at level 0, then we're done
                if(currLevel==0){
                    break;
                }

                // Specify that we're now working on the right side
                currSegment |= 1UL << currLevel;

                // Set the left and right endpoints of this level based on the previous level
                workStartInd = (2*currLevel + 5)*this->fdim_ + 2*currLevel;

                if(currLevel>0){
                    prevStartInd = (2*(currLevel-1) + 5)*this->fdim_ + 2*(currLevel-1);
                    workspace[workStartInd] = 0.5*(workspace[prevStartInd] + workspace[prevStartInd+1]); // Set the left point at the next level
                    workspace[workStartInd+1] = workspace[prevStartInd+1];
                }else{
                    workspace[workStartInd] = 0.5*(lb+ub);
                    workspace[workStartInd+1] = ub;
                }

                UpdateValues(workspace, currLevel, currSegment, leftFunc, midFunc, rightFunc);

            // If not successful, move on to the left segment at the next level
            }else{

                currLevel++;

                rightFunc = midFunc;
                midFunc = leftMidFunc;

                workStartInd = (2*currLevel + 5)*this->fdim_ + 2*currLevel;
                workspace[workStartInd] = *leftPt; // Set the left point at the next level
                workspace[workStartInd+1] = midPt; // Set the right point at the next level
            }
        }
    }

#if defined(MPART_HAS_CEREAL)
    // Define a serialize or save/load pair as you normally would
    template <class Archive>
    void save( Archive & ar ) const
    {   
        ar(cereal::base_class<RecursiveQuadratureBase<MemorySpace>>(this));
    }
    template <class Archive>
    void load( Archive & ar )
    {   
        ar(cereal::base_class<RecursiveQuadratureBase<MemorySpace>>(this));
    }

    AdaptiveSimpson(){};
#endif 

private:

    

    KOKKOS_FUNCTION void UpdateValues(double* workspace, unsigned int currLevel, unsigned int currSegment, double* &leftVal, double* &midVal, double* &rightVal) const
    {
        leftVal = &workspace[0];
        midVal = &workspace[2*this->fdim_];
        rightVal = &workspace[this->fdim_];

        for(unsigned int level=0; level<currLevel; ++level){
            // If we branch to the right, the left value will be the mid point on the coarse level
            if( (currSegment >> (level+1)) & 1U ){
                leftVal = midVal;
                midVal = &workspace[(2*level+5+1) * this->fdim_ + 2*level+2];
            }else{
                rightVal = midVal;
                midVal = &workspace[(2*level+5) * this->fdim_ + 2*level+2];
            }
        }
    }

}; // Class AdaptiveSimpsonMixin


/**
 @brief Adaptive quadrature based on applying a Clenshaw-Curtis recursively on subintervals.
 @details The points in a Clenshaw-Curtis quadrature rule with \f$2^{L}+1\f$ points are a subset of the points in a rule with \f$2^{L+1}+1\f$ points.
          This "nesting" allows us to approximate the integral at two different levels with minimal additional integrand evaluations.  Comparing the
          integral at these values gives an indication of the error in the integral approximation.  If the error is unacceptably large, we can subdivide
          the integration domain and apply Clenshaw-Curtis rules on each subinterval.   This class implements a recursive version of this process.  Two nested
          Clenshaw-Curtis rules are used to estimate the integral, and its error.  If the error is too large, then the integration domain is split into
          two equal halves, where the nested quadrature rules can again be applied.  This recursive subdivision repeats until an acceptable error level
          is reached or until a maximum number of subdivisions has occured.
 */
template<typename MemorySpace=Kokkos::HostSpace>
class AdaptiveClenshawCurtis : public RecursiveQuadratureBase<MemorySpace>{
public:
#if defined(MPART_HAS_CEREAL)
    friend class cereal::access;
#endif 

    /**
       @brief Construct a new adaptive quadrature class with specified stopping criteria.
       @param maxSub The maximum number of subintervals allowed.
       @param maxDim The maximum dimension of the integrand.
       @param[in] absTol An absolute error tolerance used to stop the adaptive integration.
       @param[in] relTol A relative error tolerance used to stop te adaptive integration.
       @param[in] errorMetric A flag specifying the type of error metric to use.
       @param[in] level The nesting level \f$L\f$.  A coarse rule with \f$2^{L}+1\f$ points will be used and a fine rule with \f$w^{L+1}+1\f$ points will be used.
     */
   inline AdaptiveClenshawCurtis(unsigned int      level,
                          unsigned int      maxSub,
                          unsigned int      maxDim,
                          double            absTol,
                          double            relTol,
                          QuadError::Type   errorMetric,
                          unsigned int      minSub=0);

    inline AdaptiveClenshawCurtis(unsigned int      level,
                           unsigned int      maxSub,
                           unsigned int      maxDim,
                           double*           workspace,
                           double            absTol,
                           double            relTol,
                           QuadError::Type   errorMetric,
                           unsigned int      minSub=0);

    KOKKOS_FUNCTION AdaptiveClenshawCurtis(Kokkos::View<double*,MemorySpace> coarsePts,
                                           Kokkos::View<double*,MemorySpace> coarseWts,
                                           Kokkos::View<double*,MemorySpace> finePts,
                                           Kokkos::View<double*,MemorySpace> fineWts,
                                           unsigned int      maxSub,
                                           unsigned int      maxDim,
                                           double*           workspace,
                                           double            absTol,
                                           double            relTol,
                                           QuadError::Type   errorMetric,
                                           unsigned int      minSub=0) : RecursiveQuadratureBase<MemorySpace>(maxSub, maxDim, GetWorkspaceSize(maxSub, maxDim), workspace, absTol, relTol, errorMetric, minSub),
                                                                                coarsePts_(coarsePts),
                                                                                coarseWts_(coarseWts),
                                                                                finePts_(finePts),
                                                                                fineWts_(fineWts)
    {
        const unsigned int level = std::log2(coarsePts.extent(0)-2);
        assert(std::pow(2,level)+1 >=3);

        ClenshawCurtisQuadrature<MemorySpace>::GetRule(std::pow(2,level)+1,  coarseWts_.data(), coarsePts_.data());
        ClenshawCurtisQuadrature<MemorySpace>::GetRule(std::pow(2,level+1)+1,  fineWts_.data(), finePts_.data());
    };


    KOKKOS_FUNCTION static unsigned int GetWorkspaceSize(unsigned int maxSub, unsigned int fdim){return (maxSub+5)*fdim + 2*maxSub;};


    KOKKOS_INLINE_FUNCTION void SetDim(unsigned int fdim){

        // If the workspace is managed internally, we need to make sure we won't required too much memory
        if(this->internalWork_.extent(0)>0)
            assert(fdim<=this->maxDim_);

        this->fdim_ = fdim;
        this->workspaceSize_=GetWorkspaceSize(this->maxSub_, fdim);
    }

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
                                   double*             res) const
    {
        assert(this->workspace_);
        Integrate(this->workspace_, f, lb, ub, res);
    }


    template<class FunctionType>
    KOKKOS_FUNCTION void Integrate(double*             workspace,
                                   FunctionType const& f,
                                   double              lb,
                                   double              ub,
                                   double*             res) const
    {
        for(unsigned int i=0; i<this->fdim_; ++i){
            res[i] = 0;
        }

        unsigned int currLevel = 0;

        unsigned int currSegment = 0;

        double midPt = 0.5*(lb+ub);
        double scale = (ub-lb);

        double* leftFunc = &workspace[0];
        f(lb, leftFunc);

        double* rightFunc = &workspace[this->fdim_];
        f(ub, rightFunc);

        double* fval = &workspace[2*this->fdim_];

        double* intCoarse = &workspace[3*this->fdim_];
        double* intFine = &workspace[4*this->fdim_];

        double* leftPt  = &workspace[5*this->fdim_];
        *leftPt = lb;
        double* rightPt  = &workspace[5*this->fdim_+1];
        *rightPt = ub;
        double* midFunc;

        unsigned int workStartInd =  5*this->fdim_;
        unsigned int prevStartInd = 0;

        double error, errorTol;

        unsigned int coarseRightIndex, coarseMidIndex, fineRightIndex, fineMidIndex;

        while(true){

            leftPt = &workspace[workStartInd];
            rightPt = &workspace[workStartInd+1];

            // Compute mid point and scale
            midPt = 0.5*((*leftPt) + (*rightPt));
            scale = 0.5*((*rightPt) - (*leftPt));

            // Evaluate the integrand at the midpoint and store in the workspace
            midFunc = &workspace[workStartInd + 2];
            f(midPt, midFunc);

            // Start with the left, right, and middle points
            coarseRightIndex = this->coarseWts_.extent(0)-1;
            coarseMidIndex = coarseRightIndex/2;
            fineRightIndex = this->fineWts_.extent(0)-1;
            fineMidIndex = fineRightIndex/2;

            // Add the
            for(unsigned int i=0; i<this->fdim_; ++i){
                intCoarse[i] =  scale*this->coarseWts_(0) * leftFunc[i];
                intCoarse[i] += scale*this->coarseWts_(coarseMidIndex) * midFunc[i];
                intCoarse[i] += scale*this->coarseWts_(coarseRightIndex) * rightFunc[i];

                intFine[i] =  scale*this->fineWts_(0) * leftFunc[i];
                intFine[i] += scale*this->fineWts_(fineMidIndex) * midFunc[i];
                intFine[i] += scale*this->fineWts_(fineRightIndex) * rightFunc[i];
            }

            // Compute the coarse integral, and the part of the fine integral coming from the nested coarse points
            for(unsigned int i=1; i<coarseMidIndex; ++i){
                f(midPt + scale*this->coarsePts_(i), fval);
                for(unsigned int j=0; j<this->fdim_; ++j){
                    intCoarse[j] += scale*this->coarseWts_(i) * fval[j];
                    intFine[j] += scale*this->fineWts_(2*i) * fval[j];
                }
            }

            for(unsigned int i=coarseMidIndex+1; i<coarseRightIndex; ++i){
                f(midPt + scale*this->coarsePts_(i), fval);
                for(unsigned int j=0; j<this->fdim_; ++j){
                    intCoarse[j] += scale*this->coarseWts_(i) * fval[j];
                    intFine[j] += scale*this->fineWts_(2*i) * fval[j];
                }
            }

            // Now add all the points that are only in the fine rule
            for (unsigned int i=1; i<this->finePts_.extent(0); i+=2){
                f(midPt + scale*this->finePts_(i), fval);
                for(unsigned int j=0; j<this->fdim_; ++j)
                    intFine[j] += scale*this->fineWts_(i) * fval[j];
            }

            // Check to see if the error is small enough or if we've hit the maximum number of subdivisions
            this->EstimateError(intCoarse, intFine, error, errorTol);

            // Checking for convergence or other termination criteria
            if((((error<errorTol)||(currLevel==this->maxSub_-1))&&(currLevel>=this->minSub_))||(std::abs(ub-lb)<1e-14)){

                for(unsigned int i=0; i<this->fdim_; ++i){
                    res[i] += intFine[i];
                }

                // Find the lowest previous level where we are working on the "left" branch of the tree.
                while(((currSegment >> currLevel) & 1U)&&(currLevel>0)){
                    currSegment &= ~(1UL << currLevel); // Set the side on this level to the left
                    currLevel--;
                }

                // If we're back at level 0, then we're done
                if(currLevel==0){
                    break;
                }

                // Specify that we're now working on the right side
                currSegment |= 1UL << currLevel;

                // Set the left and right endpoints of this level based on the previous level
                workStartInd = (currLevel + 5)*this->fdim_ + 2*currLevel;

                if(currLevel>0){
                    prevStartInd = ((currLevel-1) + 5)*this->fdim_ + 2*(currLevel-1);
                    workspace[workStartInd] = 0.5*(workspace[prevStartInd] + workspace[prevStartInd+1]); // Set the left point at the next level
                    workspace[workStartInd+1] = workspace[prevStartInd+1];
                }else{
                    workspace[workStartInd] = midPt;
                    workspace[workStartInd+1] = ub;
                }

                UpdateValues(workspace, currLevel, currSegment, leftFunc, rightFunc);

            // If not successful, move on to the left segment at the next level
            }else{

                currLevel++;

                rightFunc = midFunc;

                workStartInd = (currLevel + 5)*this->fdim_ + 2*currLevel;
                workspace[workStartInd] = *leftPt; // Set the left point at the next level
                workspace[workStartInd+1] = midPt; // Set the right point at the next level
            }


        }

    }

#if defined(MPART_HAS_CEREAL)
    // Define a serialize or save/load pair as you normally would
    template <class Archive>
    void save( Archive & ar ) const
    {   
        ar(cereal::base_class<RecursiveQuadratureBase<MemorySpace>>(this));
        ar(coarsePts_, coarseWts_, finePts_, fineWts_);
    }
    template <class Archive>
    void load( Archive & ar )
    {   
        ar(cereal::base_class<RecursiveQuadratureBase<MemorySpace>>(this));
        ar(coarsePts_, coarseWts_, finePts_, fineWts_);
    }

    AdaptiveClenshawCurtis(){};
#endif 

private:

    KOKKOS_FUNCTION void UpdateValues(double* workspace, unsigned int currLevel, unsigned int currSegment, double* &leftVal, double* &rightVal) const
    {
        leftVal = &workspace[0];
        rightVal = &workspace[this->fdim_];

        for(unsigned int level=0; level<currLevel; ++level){

            // If we branch to the right, the left value will be the mid point on the coarse level
            if( (currSegment >> (level+1)) & 1U ){
                leftVal = &workspace[(level + 5)*this->fdim_ + 2*(level+1)];
            }else{
                rightVal = &workspace[(level + 5)*this->fdim_ + 2*(level+1)];
            }
        }
    }


    Kokkos::View<double*,MemorySpace> coarsePts_, coarseWts_, finePts_, fineWts_;

}; // class AdaptiveClenshawCurtis

template<typename MemorySpace>
inline AdaptiveClenshawCurtis<MemorySpace>::AdaptiveClenshawCurtis(unsigned int      level,
                          unsigned int      maxSub,
                          unsigned int      maxDim,
                          double            absTol,
                          double            relTol,
                          QuadError::Type   errorMetric,
                          unsigned int      minSub) : RecursiveQuadratureBase<MemorySpace>(maxSub, maxDim, GetWorkspaceSize(maxSub, maxDim), absTol, relTol, errorMetric, minSub),
                                                            coarsePts_("Coarse Pts", std::pow(2,level)+1),
                                                            coarseWts_("Coarse Wts", std::pow(2,level)+1),
                                                            finePts_("Fine Pts", std::pow(2,level+1)+1),
                                                            fineWts_("Coarse Pts", std::pow(2,level+1)+1)
{
    assert(std::pow(2,level)+1 >=3);

    Kokkos::parallel_for(1, GetRuleFunctor<MemorySpace>(std::pow(2,level)+1, coarsePts_.data(), coarseWts_.data()));
    Kokkos::parallel_for(1, GetRuleFunctor<MemorySpace>(std::pow(2,level+1)+1, finePts_.data(), fineWts_.data()));
};

template<>
inline AdaptiveClenshawCurtis<Kokkos::HostSpace>::AdaptiveClenshawCurtis(unsigned int      level,
                          unsigned int      maxSub,
                          unsigned int      maxDim,
                          double            absTol,
                          double            relTol,
                          QuadError::Type   errorMetric,
                          unsigned int      minSub) : RecursiveQuadratureBase<Kokkos::HostSpace>(maxSub, maxDim, GetWorkspaceSize(maxSub, maxDim), absTol, relTol, errorMetric,minSub),
                                                            coarsePts_("Coarse Pts", std::pow(2,level)+1),
                                                            coarseWts_("Coarse Wts", std::pow(2,level)+1),
                                                            finePts_("Fine Pts", std::pow(2,level+1)+1),
                                                            fineWts_("Coarse Pts", std::pow(2,level+1)+1)
{
    assert(std::pow(2,level)+1 >=3);
    ClenshawCurtisQuadrature<Kokkos::HostSpace>::GetRule(std::pow(2,level)+1,  coarseWts_.data(), coarsePts_.data());
    ClenshawCurtisQuadrature<Kokkos::HostSpace>::GetRule(std::pow(2,level+1)+1,  fineWts_.data(), finePts_.data());
};



template<typename MemorySpace>
inline AdaptiveClenshawCurtis<MemorySpace>::AdaptiveClenshawCurtis(unsigned int      level,
                          unsigned int      maxSub,
                          unsigned int      maxDim,
                          double*           workspace,
                          double            absTol,
                          double            relTol,
                          QuadError::Type   errorMetric,
                          unsigned int      minSub) : RecursiveQuadratureBase<MemorySpace>(maxSub, maxDim, GetWorkspaceSize(maxSub, maxDim), workspace, absTol, relTol, errorMetric, minSub),
                                                            coarsePts_("Coarse Pts", std::pow(2,level)+1),
                                                            coarseWts_("Coarse Wts", std::pow(2,level)+1),
                                                            finePts_("Fine Pts", std::pow(2,level+1)+1),
                                                            fineWts_("Coarse Pts", std::pow(2,level+1)+1)
{
    assert(std::pow(2,level)+1 >=3);

    Kokkos::parallel_for(1, GetRuleFunctor<MemorySpace>(std::pow(2,level)+1, coarsePts_.data(), coarseWts_.data()));
    Kokkos::parallel_for(1, GetRuleFunctor<MemorySpace>(std::pow(2,level+1)+1, finePts_.data(), fineWts_.data()));
};

template<>
inline AdaptiveClenshawCurtis<Kokkos::HostSpace>::AdaptiveClenshawCurtis(unsigned int      level,
                          unsigned int      maxSub,
                          unsigned int      maxDim,
                          double*           workspace,
                          double            absTol,
                          double            relTol,
                          QuadError::Type   errorMetric,
                          unsigned int      minSub) : RecursiveQuadratureBase<Kokkos::HostSpace>(maxSub, maxDim, GetWorkspaceSize(maxSub, maxDim), workspace, absTol, relTol, errorMetric,minSub),
                                                            coarsePts_("Coarse Pts", std::pow(2,level)+1),
                                                            coarseWts_("Coarse Wts", std::pow(2,level)+1),
                                                            finePts_("Fine Pts", std::pow(2,level+1)+1),
                                                            fineWts_("Coarse Pts", std::pow(2,level+1)+1)
{
    assert(std::pow(2,level)+1 >=3);
    ClenshawCurtisQuadrature<Kokkos::HostSpace>::GetRule(std::pow(2,level)+1,  coarseWts_.data(), coarsePts_.data());
    ClenshawCurtisQuadrature<Kokkos::HostSpace>::GetRule(std::pow(2,level+1)+1,  fineWts_.data(), finePts_.data());
};



} // namespace mpart

#endif