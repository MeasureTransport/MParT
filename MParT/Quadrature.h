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
     @brief Approximates the integral \f$\int_{L}^{U} f(x) dx\f$ using a Clenshaw-Curtis quadrature rule.
     @param[in] f The integrand
     @param[in] lb The lower bound \f$L\f$ in the integration.
     @param[in] lb The upper bound \f$U\f$ in the integration.
     @returns A pair containing (wts,pts)
     */
    template<class ScalarFuncType>
    double Integrate(ScalarFuncType const& f, 
                    double                 lb, 
                    double                 ub)
    {   
        // Compute CC rule
        Eigen::VectorXd wts(_order), pts(_order);
        std::tie(wts,pts) = ClenshawCurtisQuadrature::GetRule(_order);

        // Rescale inputs to domain [L,U]
        pts = 0.5*(ub+lb)*Eigen::VectorXd::Ones(_order) + 0.5*(ub-lb)*pts;
        // Rescale weights
        wts = 0.5*(ub-lb)*wts;

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
 @brief Adaptive Simpson-rule integration based on applying a simple Simpson 1/3 rule recursively on subintervals.

 */
class AdaptiveSimpson {
public:

    AdaptiveSimpson(unsigned int maxSub, double absTol=1e-8, double relTol=1e-10) : _maxSub(maxSub), _absTol(absTol), _relTol(relTol), _status(0), _maxLevel(0){};
    
    /**
      @brief Approximates and integral \f$\int_{L}^{U} f(x) dx\f$
      @param[in] f The function f(x) to evaluate. The ScalarFuncType must have a call operator that accepts a single double, i.e., `operator()(double x)`.
      @param[in] lb The lower bound \f$L\f$ in the integration.
      @param[in] ub The upper bound \f$U\f$ in the integration.
     */
    template<class ScalarFuncType>
    double Integrate(ScalarFuncType const& f, 
                    double                 lb, 
                    double                 ub) {
        double integral;

        double flb = f(lb);
        double fub = f(ub);
        double midPt = 0.5*(lb+ub);
        double fmb = f(midPt);

        double intCoarse = ((ub-lb)/6.0) * (flb + 4.0*fmb + fub);
        
        std::tie(integral, _status, _maxLevel) = RecursiveIntegrate(f, lb, midPt, ub, flb, fmb, fub, 0, intCoarse);
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

    template<class ScalarFuncType>
    std::tuple<double,int, unsigned int> RecursiveIntegrate(ScalarFuncType const& f, 
                                                            double leftPt,
                                                            double midPt,
                                                            double rightPt,
                                                            double leftFunc, 
                                                            double midFunc,
                                                            double rightFunc,
                                                            int    level,
                                                            double intCoarse) {

        // update current refinement level
        level += 1;
        
        // evluate integral on each sub-interval
        double leftMidPt = 0.5*(leftPt+midPt);
        double leftMidFunc = f(leftMidPt);

        double rightMidPt = 0.5*(midPt+rightPt);
        double rightMidFunc = f(rightMidPt);

        double intFinerLeft, intFinerRight;
        intFinerLeft  = ((midPt-leftPt)/6.0) * (leftFunc + 4.0*leftMidFunc + midFunc);
        intFinerRight = ((rightPt-midPt)/6.0) * (midFunc + 4.0*rightMidFunc + rightFunc);

        // compute total integral
        double intFiner = intFinerLeft + intFinerRight;

        // Compute error and tolerance
        double intErr, tol;
        intErr = std::abs(intFiner-intCoarse);
        tol = std::fmax( _relTol*std::abs(intCoarse), _absTol);
      
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
            double intLeft, intRight;
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
      @brief Approximates and integral \f$\int_{L}^{U} f(x) dx\f$
      @param[in] f The function f(x) to evaluate. The ScalarFuncType must have a call operator that accepts a single double, i.e., `operator()(double x)`.
      @param[in] lb The lower bound \f$L\f$ in the integration.
      @param[in] ub The upper bound \f$U\f$ in the integration.
     */
    template<class ScalarFuncType>
    double Integrate(ScalarFuncType const& f, 
                    double                 lb, 
                    double                 ub) {
        double integral;
        std::tie(integral, _status, _maxLevel) = RecursiveIntegrate(f, lb, ub, 0, 0.0);
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

    template<class ScalarFuncType>
    std::tuple<double,int, unsigned int> RecursiveIntegrate(ScalarFuncType const& f, 
                                             double lb, 
                                             double ub, 
                                             int level, 
                                             double intCoarse) {

        // update current refinement level
        level += 1;

        // evaluate intCoarse on first call of function
        if (level == 1) {
            intCoarse = _quad.Integrate(f, lb, ub);
        }
        
        // evluate integral on each sub-interval
        double mb, intFinerLeft, intFinerRight;
        mb = lb+0.5*(ub-lb);
        intFinerLeft  = _quad.Integrate(f, lb, mb);
        intFinerRight = _quad.Integrate(f, mb, ub);

        // compute total integral
        double intFiner = intFinerLeft + intFinerRight;

        // Compute error and tolerance
        double intErr, tol;
        intErr = std::abs(intFiner-intCoarse);
        tol = std::fmax( _relTol*std::abs(intCoarse), _absTol);
      
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
            double intLeft, intRight;
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