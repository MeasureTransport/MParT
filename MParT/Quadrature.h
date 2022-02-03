#ifndef MPART_QUADRATURE_H
#define MPART_QUADRATURE_H

#include <math.h>
#include <sstream>
#include <Eigen/Core>

#include <iostream>

namespace mpart{

/**
@class ClenshawCurtisQuadrature
@brief Implementation of the Clenshaw-Curtis 1d quadrature rule.
@details
@code{.cpp}

// Define the integrand using a lambda function
auto f = [](double x){return exp(x);};

// Define a 5 point quadrature rule
ClenshawCurtisQuadrature quad(5);

// Integrate the function
double integral = quad.Integrate(f, 0,1);
@endcode

@code 
// Define the integrand as a class
class ExpIntegrand{
public:
    double operator()(double x){
        return exp(x);
    };
};

// Create an instance of the integrand
ExpIntegrand f;

// Define a 5 point quadrature rule
ClenshawCurtisQuadrature quad(5);

// Integrate the function
double integral = quad.Integrate(f, 0,1);

@endcode 

@code
// Get the points and weights used in the rule
Eigen::VectorXd wts, pts;
std::tie(wts, pts) = ClenshawCurtisQuadrature::GetRule(order);
@endcode

 */
class ClenshawCurtisQuadrature {
public: 

    ClenshawCurtisQuadrature(unsigned int order) : _order(order)
    {
        if(order<=0){
            std::stringstream msg;
            msg << "In MParT::RecursiveQuadrature: Quadrature order must be at least one, but given a value of \"" << order << "\".";
            throw std::runtime_error(msg.str());
        }

    };

    /**
     @brief Returns the weights and points in a Clenshaw-Curtis rule.
     @details
     @param[in] order The order of the Clenshaw-Curtis rule.
     @returns A pair containing (wts,pts)
     */
    static std::pair<Eigen::VectorXd, Eigen::VectorXd> GetRule(unsigned int order)
    {
        Eigen::VectorXd wts(order), pts(order);
        
        // return results for order == 1
        if (order == 1) {
            pts(1) = 0.0;
            wts(1) = 2.0;
            return std::make_pair(wts,pts);
        }

        // define quadrature points
        for (unsigned int i=0; i<order; ++i) {
            pts(i) = std::cos( ( order - (i+1) ) * M_PI / ( order - 1 ) );
        }
        pts(0) = -1.0;
        if ( order % 2 == 1) {
            pts((order-1)/2) = 0.0;
        }
        pts(order-1) = 1.0;

        // compute quadrature weights 
        wts.setOnes();
        for (unsigned int i=0; i<order; ++i) {            
            double theta;
            theta = i * M_PI / ( order - 1 );
            for (unsigned int j=0; j<(order-1)/2; ++j){
                double b;
                if ( 2 * (j+1) == ( order - 1 ) ) {
                    b = 1.0;
                } else {
                    b = 2.0;
                }
                wts(i) = wts(i) - b * std::cos ( 2.0 * (j+1) * theta ) / ( 4 * (j+1) * (j+1) - 1 );
            }
        }

        // adjust boundary weights
        wts(0) = wts(0) / ( order - 1.0 );
        for (unsigned int i=1; i<(order-1); ++i){
            wts(i) = 2.0 * wts(i) / ( order - 1.0 );
        }
        wts(order-1) = wts(order-1) / ( order - 1.0 );
        
        return std::make_pair(wts,pts);
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
    auto Integrate(FunctionType const& f, 
                   double              lb, 
                   double              ub) -> decltype(f(0.0))
    {   
        if(ub<lb+1e-14){
            return 0.0*f(lb);
        }

        // Compute CC rule
        Eigen::VectorXd wts,pts;
        std::tie(wts,pts) = ClenshawCurtisQuadrature::GetRule(_order);

        // Rescale inputs to domain [L,U]
        pts = 0.5*(ub+lb)*Eigen::VectorXd::Ones(_order) + 0.5*(ub-lb)*pts;

        // Rescale weights
        wts = 0.5*(ub-lb)*wts;

        // Create an output variable
        decltype(f(0.0)) output = wts[0]*f(pts[0]);

        // Evaluate integral and store results in output
        for (unsigned int i=1; i<pts.size(); ++i)
            output += wts[i] * f(pts[i]);

        return output;
    }


private:
    unsigned int _order;

}; // class ClenshawCurtisQuadrature



/**
 @brief Adaptive Simpson-rule integration based on applying a simple Simpson 1/3 rule recursively on subintervals.

 */
class AdaptiveSimpson {
public:

    AdaptiveSimpson(unsigned int maxSub, double absTol=1e-8, double relTol=1e-10) : _maxSub(maxSub), _absTol(absTol), _relTol(relTol), _status(0), _maxLevel(0){};
    
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
    auto Integrate(FunctionType const& f, 
                   double              lb, 
                   double              ub) -> decltype(f(0.0))
    { 
        auto flb = f(lb);
        auto fub = f(ub);
        double midPt = 0.5*(lb+ub);
        auto fmb = f(midPt);

        decltype(flb) intCoarse = ((ub-lb)/6.0) * (flb + 4.0*fmb + fub);
        
        decltype(flb) integral;
        std::tie(integral, _status, _maxLevel) = RecursiveIntegrate(f, lb, midPt, ub, flb, fmb, fub, 0, intCoarse);
        return integral;
    }


    // /** @brief Integrates two functions over the same domain using the same points.  
    
    //     @details This function simultaneously approximates two integrals
    //     \f\[
    //         \begin{aligned}
    //         I_1 & = \int_{x_L}^{x_U} f_1(x) dx\\
    //         I_2 & = \int_{x_L}^{x_U} f_2(x) dx.
    //         \end{aligned}
    //     \f\] 
    //     An adaptive Simpson rule is used to approximate the integrals.  While the same points will be used to 
    //     approximation \f$I_1\f$ and \f$I_2\f$, the adaptation will be driven solely by estimated errors in computing 
    //     \f$I_1\f$.   The approximation for \f$I_1\f$ is therefore the same as what would be returned by the single-integrand 
    //     `Integrate` function.   

    //     @param[in] f1 The first integrand \f$f_1\f$.
    //     @param[in] f2 The second integrand \f$f_2\f$.
    //     @return A pair of estimates \f$(\hat{I}_1, \hat{I}_2)\f$.
    //     @tparam ScalarFuncType1 The type of the first integrand.  Must provide an operator() function that accepts a single double and returns a single double.
    //     @tparam ScalarFuncType2 The type of the second integrand.  Must provide an operator() function that accepts a single double and returns a single double.
    // */
    // template<class ScalarFuncType1, class ScalarFuncType2>
    // std::pair<double,double> Integrate(ScalarFuncType1 const& f1, 
    //                                    ScalarFuncType2 const& f2,
    //                                    double                 lb, 
    //                                    double                 ub)
    // {   
    //     double integral1, integral2;

    //     double f1lb = f1(lb);
    //     double f2lb = f2(lb);

    //     double f1ub = f1(ub);
    //     double f2ub = f2(ub);

    //     double midPt = 0.5*(lb+ub);
    //     double f1mb = f1(midPt);
    //     double f2mb = f2(midPt);

    //     double intCoarse1 = ((ub-lb)/6.0) * (f1lb + 4.0*f1mb + f1ub);
    //     double intCoarse2 = ((ub-lb)/6.0) * (f2lb + 4.0*f2mb + f2ub);
        
    //     std::tie(integral1, integral2, _status, _maxLevel) = RecursiveIntegrate2(f1, f2, lb, midPt, ub, f1lb, f1mb, f1ub, f2lb, f2mb, f2ub, 0, intCoarse1, intCoarse2);
    //     return std::make_pair(integral1, integral2);
    // }

    /**
     * @brief Returns a convergence flag for the last call to "Integrate"
       
       @return The convergence flag.  Positive if successful.  Negative if not.  Zero if Integrate hasn't been called yet.
     */
    int Status() const{return _status;};

    /**
     * @brief Returns the deepest level rea
     * 
     * @return int 
     */
    int MaxLevel() const {return _maxLevel;};

private:

    template<typename IntegralValueType>
    std::pair<double, double> EstimateError(IntegralValueType const& coarseVal,
                                            IntegralValueType const& fineVal)
    {
        double error = std::abs(fineVal - coarseVal);
        double tol = std::fmax( _relTol*std::abs(coarseVal), _absTol);

        return std::make_pair(error, tol);
    }

    std::pair<double, double> EstimateError(Eigen::VectorXd const& coarseVal,
                                            Eigen::VectorXd const& fineVal)
    {
        
        double error = std::abs(fineVal(0)-coarseVal(0));
        double tol = std::fmax( _relTol*std::abs(coarseVal(0)), _absTol);

        return std::make_pair(error, tol);
    }

    template<class ScalarFuncType>
    auto RecursiveIntegrate(ScalarFuncType const& f, 
                            double leftPt,
                            double midPt,
                            double rightPt,
                            decltype(f(0.0)) leftFunc, 
                            decltype(f(0.0)) midFunc,
                            decltype(f(0.0)) rightFunc,
                            int    level,
                            decltype(f(0.0)) intCoarse) -> std::tuple<decltype(f(0.0)), int, unsigned int>
    {
        if((rightPt-leftPt)<1e-14){
            return std::make_tuple((leftPt-rightPt)*midFunc, -1, level);
        }

        // update current refinement level
        level += 1;
        
        // evluate integral on each sub-interval
        double leftMidPt = 0.5*(leftPt+midPt);
        auto leftMidFunc = f(leftMidPt);

        double rightMidPt = 0.5*(midPt+rightPt);
        auto rightMidFunc = f(rightMidPt);

        auto intFinerLeft  = ((midPt-leftPt)/6.0) * (leftFunc + 4.0*leftMidFunc + midFunc);
        auto intFinerRight = ((rightPt-midPt)/6.0) * (midFunc + 4.0*rightMidFunc + rightFunc);

        // compute total integral
        auto intFiner = intFinerLeft + intFinerRight;

        // Compute error and tolerance
        double intErr, tol;
        std::tie(intErr, tol) = EstimateError(intCoarse, intFiner);

        // Stop the recursion if the level hit maximum depth
        if ( (intErr > tol) && (level == _maxSub) ) {
            return std::make_tuple(intFiner, int(-1), level);
            //std::stringstream msg;
            //msg << "In MParT::RecursiveQuadrature: Reached maximum level depth \"" << _maxSub << "\", with an error of \"" << intErr << "\".";
            //throw std::runtime_error(msg.str());
        }
        // If the error between levels is smaller than Tol, return the finer level result
        else if ( intErr <= tol ) {
            return std::make_tuple(intFiner, int(1), level);
        }
        // Else subdivide further
        else {
            decltype(f(0.0)) intLeft, intRight;
            int statusLeft, statusRight;
            unsigned int levelLeft, levelRight;
            std::tie(intLeft, statusLeft, levelLeft) = RecursiveIntegrate(f, leftPt, leftMidPt, midPt, leftFunc, leftMidFunc, midFunc, level, intFinerLeft);
            std::tie(intRight, statusRight, levelRight) = RecursiveIntegrate(f, midPt, rightMidPt, rightPt, midFunc, rightMidFunc, rightFunc, level, intFinerRight);
            
            return std::make_tuple(intLeft + intRight, int( ((statusLeft<0)||(statusLeft<0))?-1:1 ), std::max(levelLeft, levelRight)); 
        }

    }

    const unsigned int _maxSub;
    const double _absTol;
    const double _relTol;

    int _status;
    unsigned int _maxLevel;

}; // Class AdaptiveSimpson



/**
 @brief Adaptive quadrature based on applying a Clenshaw-Curtis recursively on subintervals.

 */
class RecursiveQuadrature {
public:

    /**
       @brief Construct a new adaptive quadrature class with specified stopping criteria.
       @param maxSub The maximum number of subintervals allowed.
       @param[in] order Number of points to use per subinterval.
       @param[in] absTol An absolute error tolerance used to stop the adaptive integration.
       @param[in] relTol A relative error tolerance used to stop te adaptive integration.
     */
    RecursiveQuadrature(unsigned int maxSub, unsigned int order=2, double absTol=1e-8, double relTol=1e-10) : _maxSub(maxSub), _order(order), _absTol(absTol), _relTol(relTol), _quad(order), _status(0), _maxLevel(0)
    {
        if(absTol<=0){
            std::stringstream msg;
            msg << "In MParT::RecursiveQuadrature: Absolute error tolerance must be strictly positive, but given a value of \"" << absTol << "\".";
            throw std::runtime_error(msg.str());
        }
        if(relTol<=0){
            std::stringstream msg;
            msg << "In MParT::RecursiveQuadrature: Relative error tolerance must be strictly positive, but given a value of \"" << relTol << "\".";
            throw std::runtime_error(msg.str());
        }
        if(maxSub==0){
            std::stringstream msg;
            msg << "In MParT::RecursiveQuadrature: Maximum subintervals allowed must be greater than 0, but given a value of \"" << maxSub << "\".";
            throw std::runtime_error(msg.str());
        }

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
    auto Integrate(FunctionType const& f, 
                   double              lb, 
                   double              ub) -> decltype(f(0.0))
    {   
        auto intCoarse = _quad.Integrate(f, lb, ub);
        decltype(f(0.0)) integral;
        std::tie(integral, _status, _maxLevel) = RecursiveIntegrate(f, lb, ub, 0, intCoarse);
        return integral;
    }


    /**
     * @brief Returns a convergence flag for the last call to "Integrate"
       
       @return The convergence flag.  Positive if successful.  Negative if not.  Zero if Integrate hasn't been called yet.
     */
    int Status() const{return _status;};

    /**
     * @brief Returns the deepest level rea
     * 
     * @return int 
     */
    int MaxLevel() const {return _maxLevel;};

private:

    template<typename IntegralValueType>
    std::pair<double, double> EstimateError(IntegralValueType const& coarseVal,
                                            IntegralValueType const& fineVal)
    {
        double error = std::abs(fineVal - coarseVal);
        double tol = std::fmax( _relTol*std::abs(coarseVal), _absTol);

        return std::make_pair(error, tol);
    }

    std::pair<double, double> EstimateError(Eigen::VectorXd const& coarseVal,
                                            Eigen::VectorXd const& fineVal)
    {
        
        double error = std::abs(fineVal(0)-coarseVal(0));
        double tol = std::fmax( _relTol*std::abs(coarseVal(0)), _absTol);

        return std::make_pair(error, tol);
    }

    template<class FunctionType>
    auto RecursiveIntegrate(FunctionType const& f, 
                            double lb, 
                            double ub, 
                            int level, 
                            decltype(f(0.0)) intCoarse) -> std::tuple<decltype(f(0.0)),int, unsigned int>
    {

        // update current refinement level
        level += 1;

        // evluate integral on each sub-interval
        double mb = lb+0.5*(ub-lb);
        auto intFinerLeft  = _quad.Integrate(f, lb, mb);
        auto intFinerRight = _quad.Integrate(f, mb, ub);

        // compute total integral
        decltype(intCoarse) intFiner = intFinerLeft + intFinerRight;

        // Compute error and tolerance
        double intErr, tol;
        std::tie(intErr, tol) = EstimateError(intCoarse, intFiner);
        
        // Stop the recursion if the level hit maximum depth
        if ( (intErr > tol) && (level == _maxSub) ) {
            return std::make_tuple(intFiner, int(-1), level);
            //std::stringstream msg;
            //msg << "In MParT::RecursiveQuadrature: Reached maximum level depth \"" << _maxSub << "\", with an error of \"" << intErr << "\".";
            //throw std::runtime_error(msg.str());
        }
        // If the error between levels is smaller than Tol, return the finer level result
        else if ( intErr <= tol ) {
            return std::make_tuple(intFiner, int(1), level);
        }
        // Else subdivide further
        else {
            decltype(intCoarse) intLeft, intRight;
            int statusLeft, statusRight;
            unsigned int levelLeft, levelRight;
            std::tie(intLeft, statusLeft, levelLeft) = RecursiveIntegrate(f, lb, mb, level, intFinerLeft);
            std::tie(intRight, statusRight, levelRight) = RecursiveIntegrate(f, mb, ub, level, intFinerRight);
            
            return std::make_tuple(intLeft + intRight, int( ((statusLeft<0)||(statusLeft<0))?-1:1 ), std::max(levelLeft, levelRight)); 
        }

    }

private:
    const unsigned int _maxSub;
    const unsigned int _order;
    const double _absTol;
    const double _relTol;

    ClenshawCurtisQuadrature _quad;

    // Saved convergence diagnostics
    int _status;
    unsigned _maxLevel;

}; // class RecursiveQuadrature


} // namespace mpart

#endif 