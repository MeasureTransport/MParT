#ifndef MPART_QUADRATURE_H
#define MPART_QUADRATURE_H

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <tuple>

#include <Eigen/Core>

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


class RecursiveQuadratureBase
{

public:

    RecursiveQuadratureBase(unsigned int maxSub,
                            double absTol,
                            double relTol,
                            QuadError::Type errorMetric) :  _maxSub(maxSub),
                                                            _absTol(absTol),
                                                            _relTol(relTol),
                                                            _status(0),
                                                            _maxLevel(0),
                                                            _errorMetric(errorMetric)
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
    };

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


protected:
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
        double error, tol, relRefVal;
        if(_errorMetric==QuadError::First){
            error = std::abs(fineVal(0)-coarseVal(0));
            relRefVal = std::abs(coarseVal(0));
        }else if(_errorMetric==QuadError::NormInf){
            error = (fineVal-coarseVal).array().abs().maxCoeff();
            relRefVal = coarseVal.array().abs().maxCoeff();
        }else if(_errorMetric==QuadError::Norm2){
            error = (fineVal-coarseVal).norm();
            relRefVal = coarseVal.norm();
        }else if(_errorMetric==QuadError::Norm1){
            error = (fineVal-coarseVal).array().abs().sum(); //std::abs(fineVal(0)-coarseVal(0));
            relRefVal = coarseVal.array().abs().sum();
        }
        tol = std::fmax( _relTol*relRefVal, _absTol);
        return std::make_pair(error, tol);
    }

    const unsigned int _maxSub;
    const double _absTol;
    const double _relTol;

    int _status;
    unsigned int _maxLevel;
    QuadError::Type _errorMetric;
};



/**
 @brief Adaptive Simpson-rule integration based on applying a simple Simpson 1/3 rule recursively on subintervals.

 */
class AdaptiveSimpson : public RecursiveQuadratureBase {
public:

    AdaptiveSimpson(unsigned int maxSub,
                    double absTol,
                    double relTol,
                    QuadError::Type errorMetric) : RecursiveQuadratureBase(maxSub, absTol, relTol, errorMetric){};

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

private:
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

}; // Class AdaptiveSimpsonMixin


/**
 @brief Adaptive quadrature based on applying a Clenshaw-Curtis recursively on subintervals.
 */
class AdaptiveClenshawCurtis : public RecursiveQuadratureBase{
public:

    /**
       @brief Construct a new adaptive quadrature class with specified stopping criteria.
       @param maxSub The maximum number of subintervals allowed.
       @param[in] absTol An absolute error tolerance used to stop the adaptive integration.
       @param[in] relTol A relative error tolerance used to stop te adaptive integration.
       @param[in] errorMetric A flag specifying the type of error metric to use.
       @param[in] order Number of points to use per subinterval.
     */
    AdaptiveClenshawCurtis(unsigned int maxSub,
                    double              absTol,
                    double              relTol,
                    QuadError::Type     errorMetric,
                    unsigned int        order) : RecursiveQuadratureBase(maxSub, absTol, relTol, errorMetric),
                                                 _order(order),
                                                 _quad(order){};

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

private:

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

    const unsigned int _order;
    ClenshawCurtisQuadrature _quad;

}; // class AdaptiveClenshawCurtis


} // namespace mpart

#endif