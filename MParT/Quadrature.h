#ifndef MPART_QUADRATURE_H
#define MPART_QUADRATURE_H

#include <sstream>
//#include <Eigen/Core>

namespace mpart{

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
class ClenshawCurtisQuadrature
{
    ClenshawCurtisQuadrature(unsigned int order) : _order(order)
    {
    };

    /**
     @brief Returns the weights and points in a Clenshaw-Curtis rule.
     @param[in] order The order of the Clenshaw-Curtis rule.
     @returns A pair containing (wts,pts)
     */
    //static std::pair<Eigen::VectorXd, Eigen::VectorXd> GetRule(unsigned int order);
    
    
    template<class ScalarFuncType>
    double Integrate(ScalarFuncType const& f, 
                    double                 lb, 
                    double                 ub)
    {   
        return 1.0;
    }

private:
    unsigned int _order;

}; // class ClenshawCurtisQuadrature

/**
 @brief Adaptive quadrature based on applying a Clenshaw-Curtis recursively on subintervals.

 */
class RecursiveQuadrature {
public:

    /**
       @brief Construct a new adaptive quadrature class with specified stopping criteria.
       @param maxSub The maximum number of subintervals allowed.
       @param[in] absTol An absolute error tolerance used to stop the adaptive integration.
       @param[in] relTol A relative error tolerance used to stop te adaptive integration.
     */
    RecursiveQuadrature(unsigned int maxSub, double absTol=1e-8, double relTol=1e-10) : _maxSub(maxSub), _absTol(absTol), _relTol(relTol)
    {
        if(absTol<=0){
            std::stringstream msg;
            msg << "In MParT::GaussKronrad: Absolute error tolerance must be strictly positive, but given a value of \"" << absTol << "\".";
            throw std::runtime_error(msg.str());
        }
        if(relTol<=0){
            std::stringstream msg;
            msg << "In MParT::GaussKronrad: Relative error tolerance must be strictly positive, but given a value of \"" << relTol << "\".";
            throw std::runtime_error(msg.str());
        }
        
        if(maxSub==0){
            std::stringstream msg;
            msg << "In MParT::GaussKronrad: Maximum subintervals allowed must be greater than 0, but given a value of \"" << maxSub << "\".";
            throw std::runtime_error(msg.str());
        }
    }

    /**
      @brief Approximates and integral \f$\int_{L}^{U} f(x) dx\f$

      @param[in] f The function f(x) to evaluate. The ScalarFuncType must have a call operator that accepts a single double, i.e., `operator()(double x)`.
      @param[in] lb The lower bound \f$L\f$ in the integration.
      @param[in] ub The upper bound \f$U\f$ in the integration.
     */
    template<class ScalarFuncType>
    double Integrate(ScalarFuncType const& f, 
                    double                 lb, 
                    double                 ub)
    {
        return 1.0;
    }

private:
    unsigned int _maxSub;
    double _absTol;
    double _relTol;

}; // class GaussKonrad


} // namespace mpart

#endif 