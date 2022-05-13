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


class QuadratureBase {

public:

    KOKKOS_FUNCTION QuadratureBase(unsigned int maxDim, 
                                   unsigned int workspaceSize) : fdim_(maxDim),
                                                                 maxDim_(maxDim),
                                                                 workspaceSize_(workspaceSize),
                                                                 internalWorkspace_(workspaceSize),
                                                                 workspace_(&internalWorkspace_[0])
    {
    }

    KOKKOS_FUNCTION QuadratureBase(unsigned int maxDim, 
                                   unsigned int workspaceSize,
                                   double*      workspace) : fdim_(maxDim),
                                                             maxDim_(maxDim),
                                                             workspaceSize_(workspaceSize),
                                                             workspace_(workspace)
    {
    }

    KOKKOS_INLINE_FUNCTION void SetWorkspace(double* workspace){workspace_ = workspace;};
    KOKKOS_INLINE_FUNCTION void SetDim(unsigned int fdim){assert(fdim<=maxDim_); fdim_ = fdim;}


protected:

    unsigned int fdim_; // The current length of f(x)
    const unsigned int maxDim_; // The maximum length of f(x) allowed by the cache size
    const unsigned int workspaceSize_; // The total number of doubles that could be used by this quadrature method to store function evaluations during a call to "integrate"

    std::vector<double> internalWorkspace_;
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
class ClenshawCurtisQuadrature {
public: 

    /**
     * @brief Construct a new Clenshaw Curtis Quadrature object given the number of points and maximum integrand dimension
     * 
     * @param numPts The number of points in the CC rule.
     * @param maxDim The maximum dimension of the integrand.  Used to help set up workspace.
     */
    KOKKOS_FUNCTION ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim) : QuadratureBase(maxDim,maxDim), numPts_(numPts)
    {
        GetRule(numPts, &workspace_[0], &workspace_[numPts]);
    };

    /**
     * @brief Construct a new Clenshaw Curtis Quadrature object with a predefined workspace.
     * 
     * @param numPts The number of points in the CC rule.
     * @param maxDim The maximum dimension of the integrand.  Used to help set up workspace.
     * @param workspace A pointer to memory that is allocated as a workspace.  Must have space for at least 2*numPts+maxDim components.
     */

    KOKKOS_FUNCTION ClenshawCurtisQuadrature(unsigned int numPts, unsigned int maxDim, double* workspace) : QuadratureBase(maxDim,2*numPts, workspace), numPts_(numPts)
    {
        GetRule(numPts, &workspace_[0], &workspace_[numPts]);
    };

    /**
     @brief Returns the weights and points in a Clenshaw-Curtis rule.
     @param[in] order The order of the Clenshaw-Curtis rule.
     @returns A pair containing (wts,pts)
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
        for (unsigned int i=0; i<order; ++i) {
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
            wts(i) = 2.0 * wts[i] / ( numPts - 1.0 );
        }
        wts[numPts-1] = wts[numPts-1] / ( numPts - 1.0 );
    }
    
    /**
     @brief Approximates the integral \f$\int_{x_L}^{x_U} f(x) dx\f$ using a Clenshaw-Curtis quadrature rule.
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
        if((ub-lb)<15.0*std::numeric_limits<double>::epsilon()){

            f(0.5*(ub+lb), res);
            for(unsigned int i=0; i<fdim_; ++i)
                res[i] *= (ub-lb);
                
            return;
        }

        // // Rescale inputs to domain [L,U]
        // pts = 0.5*(ub+lb)*Eigen::VectorXd::Ones(_order) + 0.5*(ub-lb)*pts;

        // // Rescale weights
        // wts = 0.5*(ub-lb)*wts;

        // Create an output variable
        double* fval = &workspace_[2*numPts_];
        
        // Evaluate integral and store results in output
        for (unsigned int i=0; i<pts.size(); ++i){
            f(0.5*(ub+lb + (ub-lb)*pts[i]), fval);
            for(unsigned int j=0; j<fdim_; ++j)
                res[j] += 0.5*(ub-lb)*wts[i] * f[j];
        }

        return output;
    }

private:
    const unsigned int numPts_;

}; // class ClenshawCurtisQuadrature


class RecursiveQuadratureBase : public QuadratureBase
{

public:
    
    KOKKOS_FUNCTION RecursiveQuadratureBase(unsigned int maxSub, 
                                            unsigned int maxDim,
                                            unsigned int ptsPerSub, 
                                            double absTol, 
                                            double relTol,
                                            QuadError::Type errorMetric) :  QuadratureBase(maxDim, maxSub*ptsPerSub*maxDim),
                                                                            _maxSub(maxSub), 
                                                                            _absTol(absTol),
                                                                            _relTol(relTol),
                                                                            _errorMetric(errorMetric)
    {}

    KOKKOS_FUNCTION RecursiveQuadratureBase(unsigned int maxSub, 
                                            unsigned int maxDim,
                                            unsigned int ptsPerSub,
                                            double*      workspace, 
                                            double       absTol, 
                                            double       relTol,
                                            QuadError::Type errorMetric) :  QuadratureBase(maxDim, maxSub*ptsPerSub*maxDim, workspace),
                                                                            _maxSub(maxSub), 
                                                                            _absTol(absTol),
                                                                            _relTol(relTol),
                                                                            _errorMetric(errorMetric)
    {}


protected:

    KOKKOS_FUNCTION void EstimateError(const double* coarseVal,
                                       const double* fineVal,
                                       double      & error,
                                       double      & tol) const
    {   
        double relRefVal;
        if(_errorMetric==QuadError::First){   
            error = fabs(fineVal[0]-coarseVal[0]);
            relRefVal = fabs(coarseVal[0]);
        }else if(_errorMetric==QuadError::NormInf){
            error = 0;
            relRefVal = 0;
            for(unsigned int i=0; i<fdim_; ++i){
                error = fmax(error, fabs(fineVal[i]-coarseVal[i]));
                relRefVal = fmax(relRefVal, fabs(coarseVal[i]));
            }

        }else if(_errorMetric==QuadError::Norm2){
            
            error = 0;
            relRefVal = 0;
            for(unsigned int i=0; i<fdim_; ++i){
                error += (fineVal[i]-coarseVal[i])*(fineVal[i]-coarseVal[i]);
                relRefVal += coarseVal[i]*coarseVal[i];
            }
            error = sqrt(error);
            relRefVal = sqrt(relRefVal);

        }else if(_errorMetric==QuadError::Norm1){

            error = 0;
            relRefVal = 0;
            for(unsigned int i=0; i<fdim_; ++i){
                error += fabs(fineVal[i]-coarseVal[i]);
                relRefVal += fabs(coarseVal[i]);
            }

        }
        tol = std::fmax( _relTol*relRefVal, _absTol);
    }

    const unsigned int _maxSub;
    const double _absTol;
    const double _relTol;

    //int _status;
    //unsigned int _maxLevel;
    QuadError::Type _errorMetric;
};



/**
 @brief Adaptive Simpson-rule integration based on applying a simple Simpson 1/3 rule recursively on subintervals.

 */
class AdaptiveSimpson : public RecursiveQuadratureBase {
public:

    KOKKOS_FUNCTION AdaptiveSimpson(unsigned int maxSub,
                                    unsigned int fdim, 
                                    double absTol, 
                                    double relTol,
                                    QuadError::Type errorMetric) : RecursiveQuadratureBase(maxSub, fdim, 4, absTol, relTol, errorMetric){};

     KOKKOS_FUNCTION AdaptiveSimpson(unsigned int maxSub,
                                    unsigned int fdim, 
                                    double* workspace,
                                    double absTol, 
                                    double relTol,
                                    QuadError::Type errorMetric) : RecursiveQuadratureBase(maxSub, fdim, 4, workspace, absTol, relTol, errorMetric){};


    /** Returns the maximum number of doubles required by this quadrature rule.  Used to preallocate memory. */
    KOKKOS_INLINE_FUNCTION static unsigned int WorkspaceRequirement(unsigned int fdim, unsigned int maxSub){return 4*maxSub*fdim;}

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
        auto flb = &workspace_[0];
        auto fub = &workspace_[fdim_];
        auto fmb = &workspace_[2*fdim_];
        auto intCoarse = &workspace_[3*fdim_];

        f(lb, flb);
        f(ub, fub);

        double midPt = 0.5*(lb+ub);
        f(midPt, fmb);
        
        // Compute ((ub-lb)/6.0) * (flb + 4.0*fmb + fub);
        for(unsigned int i=0; i<fdim_; ++i) 
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

            for(unsigned int i=0; i<fdim_; ++i)
                integral[i] = (rightPt-leftPt)*midFunc[i];
            status = -1;
            levelOut = level;
            return;
        }

        // update current refinement level
        level += 1;
        
        // evaluate integral on each sub-interval
        double* leftMidFunc = &workspace_[(4*level) * fdim_];
        double* rightMidFunc = &workspace_[(4*level+1) * fdim_];
        double* intFinerLeft = &workspace_[(4*level+2) * fdim_];
        double* intFinerRight = &workspace_[(4*level+3) * fdim_];

        double leftMidPt = 0.5*(leftPt+midPt);
        f(leftMidPt, leftMidFunc);

        double rightMidPt = 0.5*(midPt+rightPt);
        f(rightMidPt, rightMidFunc);

        for(unsigned int i=0; i<fdim_; ++i){
            intFinerLeft[i]  = ((midPt-leftPt)/6.0) * (leftFunc[i] + 4.0*leftMidFunc[i] + midFunc[i]);
            intFinerRight[i] = ((rightPt-midPt)/6.0) * (midFunc[i] + 4.0*rightMidFunc[i] + rightFunc[i]);
            integral[i] = intFinerLeft[i] + intFinerRight[i];
        }

        // Compute error and tolerance
        double intErr, tol;
        EstimateError(intCoarse, integral, intErr, tol);

        // Stop the recursion if the level hit maximum depth
        if ( (intErr > tol) && (level == _maxSub) ) {
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
            for(unsigned int i=0; i<fdim_; ++i)
                integral[i] += intFinerLeft[i];

            status = int( ((statusLeft<0)||(statusLeft<0))?-1:1 );
            levelOut =  fmax(levelLeft, levelRight);
            return; 
        }
    }

}; // Class AdaptiveSimpsonMixin


// /**
//  @brief Adaptive quadrature based on applying a Clenshaw-Curtis recursively on subintervals.
//  */
// class AdaptiveClenshawCurtis : public RecursiveQuadratureBase{
// public:

//     /**
//        @brief Construct a new adaptive quadrature class with specified stopping criteria.
//        @param maxSub The maximum number of subintervals allowed.
//        @param[in] absTol An absolute error tolerance used to stop the adaptive integration.
//        @param[in] relTol A relative error tolerance used to stop te adaptive integration.
//        @param[in] errorMetric A flag specifying the type of error metric to use.
//        @param[in] order Number of points to use per subinterval.
//      */
//     AdaptiveClenshawCurtis(unsigned int maxSub, 
//                     double              absTol, 
//                     double              relTol,
//                     QuadError::Type     errorMetric,
//                     unsigned int        order) : RecursiveQuadratureBase(maxSub, absTol, relTol, errorMetric),
//                                                  _order(order), 
//                                                  _quad(order){};

//     /**
//      @brief Approximates the integral \f$\int_{x_L}^{x_U} f(x) dx\f$
//      @details
//      @param[in] f The integrand.  Can return any type that overloads multiplication with a double and the += operator.  doubles and Eigen::VectorXd are examples.
//      @param[in] lb The lower bound \f$x_L\f$ in the integration.
//      @param[in] lb The upper bound \f$x_U\f$ in the integration.
//      @returns An approximation of \f$\int_{x_L}^{x_U} f(x) dx\f$.  The return type will be the same as the type returned by the integrand function.
//      @tparam FunctionType The type of the integrand.  Must have an operator()(double x) function.
//      */
//     template<typename OutputType, class FunctionType>
//     KOKKOS_FUNCTION OutputType Integrate(FunctionType const& f, 
//                    double              lb, 
//                    double              ub) const
//     {   
//         auto intCoarse = _quad.Integrate<OutputType>(f, lb, ub);
//         OutputType integral;
//         int status;
//         unsigned int maxLevel;
//         RecursiveIntegrate<OutputType>(f, lb, ub, 0, intCoarse, integral, status, maxLevel);
//         return integral;
//     }

// private:

//     template<typename OutputType, class FunctionType>
//     KOKKOS_FUNCTION void RecursiveIntegrate(FunctionType const& f, 
//                             double lb, 
//                             double ub, 
//                             int level, 
//                             OutputType intCoarse,
//                             OutputType &integral, 
//                             int       &status, 
//                             unsigned int &levelOut) const
//     {

//         // update current refinement level
//         level += 1;

//         // evluate integral on each sub-interval
//         double mb = lb+0.5*(ub-lb);
//         auto intFinerLeft  = _quad.Integrate<OutputType>(f, lb, mb);
//         auto intFinerRight = _quad.Integrate<OutputType>(f, mb, ub);

//         // compute total integral
//         decltype(intCoarse) intFiner = intFinerLeft + intFinerRight;

//         // Compute error and tolerance
//         double intErr, tol;
//         EstimateError(intCoarse, intFiner, intErr, tol);
        
//         // Stop the recursion if the level hit maximum depth
//         if ( (intErr > tol) && (level == _maxSub) ) {
//             integral = intFiner;
//             status = -1;
//             levelOut = level;
//             return;
//         }
//         // If the error between levels is smaller than Tol, return the finer level result
//         else if ( intErr <= tol ) {
//             integral = intFiner;
//             status = 1;
//             levelOut = level;
//             return;
//         }
//         // Else subdivide further
//         else {
//             OutputType intLeft, intRight;
//             int statusLeft, statusRight;
//             unsigned int levelLeft, levelRight;
//             RecursiveIntegrate<OutputType>(f, lb, mb, level, intFinerLeft, intLeft, statusLeft, levelLeft);
//             RecursiveIntegrate<OutputType>(f, mb, ub, level, intFinerRight, intRight, statusRight, levelRight);
            
//             integral = intLeft + intRight;
//             status = int( ((statusLeft<0)||(statusLeft<0))?-1:1 );
//             levelOut = std::max(levelLeft, levelRight);
//             return;
//         }

//     }

// private:

//     const unsigned int _order;
//     ClenshawCurtisQuadrature _quad;

// }; // class AdaptiveClenshawCurtis


} // namespace mpart

#endif 