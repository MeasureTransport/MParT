#ifndef MPART_QUADRATURE_H
#define MPART_QUADRATURE_H

#include <math.h>
#include <sstream>
#include <Eigen/Core>

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
class ClenshawCurtisQuadrature {
public: 

    ClenshawCurtisQuadrature(unsigned int order) : _order(order)
    {
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
    
    template<class ScalarFuncType>
    double Integrate(ScalarFuncType const& f, 
                    double                 lb, 
                    double                 ub)
    {   
        // Compute CC rule
        Eigen::VectorXd wts(_order), pts(_order);
        std::tie(wts,pts) = ClenshawCurtisQuadrature::GetRule(_order);

        // Rescale inputs!

        // Evaluate integral
        double integral;
        integral = 0;
        for (unsigned int i=0; i<_order; ++i) {
            integral = integral + f(pts[i]) * wts[i];
        }
        return integral;
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
        //ClenshawCurtisQuadrature quad(5);
        double integral;
        //integral = quad.Integrate(f,lb,ub);
        integral = 1.0;
        return integral;
    }

private:
    unsigned int _maxSub;
    double _absTol;
    double _relTol;

}; // class RecursiveQuadrature


} // namespace mpart

#endif 