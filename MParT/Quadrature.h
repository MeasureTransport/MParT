#ifndef MPART_QUADRATURE_H
#define MPART_QUADRATURE_H

#include <sstream>

namespace mpart{

/**
 @brief Adaptive GaussKronrad quadrature.

 Adapted from https://github.com/tbs1980/NumericalIntegration/blob/master/Integrator.h and the fortran quadpack package.
 */
class GaussKronrad {
public:

    /**
     * @brief Construct a new adaptive Gauss-Kronrad quadrature class with specified stopping criteria.
     * @param maxSub The maximum number of subintervals allowed.
     * @param absTol The absolute error criteria.
     * @param relTol The relative error criteria.
     */
    GaussKronrad(unsigned int maxSub, double absTol, double relTol) : _maxSub(maxSub), _absTol(absTol), _relTol(relTol)
    {
        if(absTol<=0){
            std::stringstream msg;
            msg << "In MParT::GaussKronrad: Absolute error tolerance must be strictly positive, but given a value of \"" << absTol << "\"."
            throw std::runtime_error(msg.str());
        }
        if(relTol<=0){
            std::stringstream msg;
            msg << "In MParT::GaussKronrad: Relative error tolerance must be strictly positive, but given a value of \"" << relTol << "\"."
            throw std::runtime_error(msg.str());
        }
        
        if(maxSub==0){
            std::stringstream msg;
            msg << "In MParT::GaussKronrad: Maximum subintervals allowed must be greater than 0, but given a value of \"" << maxSub << "\"."
            throw std::runtime_error(msg.str());
        }
    }

    /**
      @brief Approximates and integral \f$\int_{L}^{U} f(x) dx\f$

      @param[in] f The function f(x) to evaluate. The ScalarFuncType must have a call operator that accepts a single double, i.e., `operator()(double x)`.
      @param[in] lb The lower bound \f$L\f$ in the integration.
      @param[in] ub The upper bound \f$U\f$ in the integration.
      @param[in] absTol An absolute error tolerance used to stop the adaptive integration.
      @param[in] relTol A relative error tolerance used to stop te adaptive integration.
     */
    template<class ScalarFuncType>
    double Integrate(ScalarFuncType const& f, 
                    double                 lb, 
                    double                 ub, 
                    double                 absTol,
                    double                 relTol)
    {
        
    }

private:
    unsigned int _maxSub;
    double _absTol;
    double _relTol;

}; // class GaussKonrad


} // namespace mpart

#endif 