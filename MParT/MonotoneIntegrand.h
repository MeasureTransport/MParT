#ifndef MPART_MONOTONEINTEGRAND_H
#define MPART_MONOTONEINTEGRAND_H

#include "MParT/DerivativeFlags.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>


namespace mpart{

/**
    @brief Computes the integrand  \f$g( \partial_d f(x_1,x_2,\ldots,x_{d-1},t) )\f$ used in a monotone component.
      
    This class assumes f is given through an expansion containing a tensor product basis functions
    \f[
      f(x_1,x_2,\ldots,x_d) = \sum_\alpha c_{\alpha\in\mathcal{A}} \prod_{d=1^D} \phi_{\alpha_d}(x_d),
    \f]
    where \f$\alpha\f$ is a multiindex in some set \f$\mathcal{A}\f$ and \f$\phi_{\alpha_d}\f$ is a univariate 
    function with degree (or general index) \f$\alpha_d\f$.   The BasisEvaluatorType template argument defines 
    the family of \f$\phi_{\alpha_d}\f$ functions.  When \f$\phi_{\alpha_d}\f$ is an orthongal polynomial, 
    it is possible to efficiently evaluate all degrees less than \f$P\f$ using the three term recurrence 
    relationship of the polynomial.  This is leveraged here to accelerate the evaluation of the integrand.  The 
    components $x_1,x_2,\ldots,x_{d-1}$ are all known apriori, as are the maximum degrees in each of those 
    directions.   The values of \f$\phi_{\alpha_d}(x_d)\f$ for \f$d<D\f$ can thus be precomputed and reused 
    during the integration of \f$g( f(x_1,x_2,\ldots,x_{d-1},t) )\f$.
    
    After the constructor has been called, cache[startPos[d]][p] will contain \phi_p(x_d).

    Note that some private member variables are stored by reference.  The user of this function must make sure that the
    arguments given to the constructor persist longer than the life of this class.
 
   @tparam BasisEvaluatorType A class defining the family of 1d basis functions used to parameterize the function \f$f\f$.  The MParT::HermiteFunction and MParT::ProbabilistHermite classes are examples of types that implement the required interface.
   @tparam PosFuncType A class defining the function \f$g\f$.  This class must have `Evaluate` and `Derivative` functions accepting a double and returning a double.  The MParT::SoftPlus and MParT::Exp classes in PositiveBijectors.h are examples of classes defining this interface.
   @tparam PointType The type of array used to store the point.  Should be some form of Kokkos::View<double*>.
   @tparam CoeffsType The type of array used to store the coeffs.  Should be some form of Kokkos::View<double*>.
 */
template<class ExpansionType, class PosFuncType, class PointType, class CoeffsType>
class MonotoneIntegrand{
public:

    

    /**
      @param cache A pointer to memory storing evaluations of \phi_{i,p}(x_i) for each i.  These terms need 
                   to be evaluated outside this class (e.g., using `_expansion.FillCache1` for \f$i\in\{0,\ldots,D-1\}\f$. 
      @param expansion 
      @param xd 
      @param coeffs
      @param derivType
     */
    MonotoneIntegrand(double*                            cache,
                      ExpansionType               const& expansion,
                      PointType                   const& pt,
                      CoeffsType                  const& coeffs,
                      DerivativeFlags::DerivativeType    derivType) : _dim(pt.extent(0)),
                                                                      _cache(cache),
                                                                      _expansion(expansion),
                                                                      _pt(pt),
                                                                      _xd(pt(_dim-1)),
                                                                      _coeffs(coeffs),
                                                                      _derivType(derivType)
    {   
    }

    MonotoneIntegrand(double*                            cache,
                      ExpansionType               const& expansion,
                      PointType                   const& pt,
                      double                             xd,
                      CoeffsType                  const& coeffs,
                      DerivativeFlags::DerivativeType    derivType) : _dim(pt.extent(0)),
                                                                      _cache(cache),
                                                                      _expansion(expansion),
                                                                      _pt(pt),
                                                                      _xd(xd),
                                                                      _coeffs(coeffs),
                                                                      _derivType(derivType)
    {   
    }


    /**
     Evaluates \f$g( \partial_d f(x_1,x_2,\ldots, x_d*t))\f$ using the cached values of \f$x\f$ given to the constructor 
     and the value of \f$t\f$ passed to this function.  Note that we assume t ranges from [0,1].  The change of variables to x_d*t is
     taken care of inside this function.
    */
    Eigen::VectorXd operator()(double t) const
    {   
        const unsigned int numTerms = _expansion.NumCoeffs();
        
        unsigned int numOutputs = 1;
        if(_derivType==DerivativeFlags::Diagonal)
            numOutputs++;
        if((_derivType==DerivativeFlags::Parameters) || (_derivType==DerivativeFlags::Mixed))
            numOutputs += numTerms;

        Eigen::VectorXd output = Eigen::VectorXd::Zero(numOutputs);


        // Finish filling in the cache at the quadrature point (FillCache1 is called outside this class)
        if((_derivType==DerivativeFlags::Diagonal)||(_derivType==DerivativeFlags::Mixed)){
            _expansion.FillCache2(_cache, _pt, t*_xd, DerivativeFlags::Diagonal2);
        }else{
            _expansion.FillCache2(_cache, _pt, t*_xd, DerivativeFlags::Diagonal);
        }
        
        // Use the cache to evaluate \partial_d f and, optionally, the gradient of \partial_d f wrt the coefficients.
        double df = 0;
        if(_derivType==DerivativeFlags::Parameters){
            Eigen::Ref<Eigen::VectorXd> gradSeg(output.tail(numTerms));
            df = _expansion.MixedDerivative(_cache, _coeffs, 1, gradSeg);
            output *= _xd*PosFuncType::Derivative(df);

        }else if(_derivType==DerivativeFlags::Mixed){
            df = _expansion.DiagonalDerivative(_cache, _coeffs, 1);

            Eigen::Ref<Eigen::VectorXd> gradSeg(output.tail(numTerms));
            Eigen::VectorXd temp(numTerms);

            double dgdf = PosFuncType::Derivative(df);
            double df2 = _expansion.MixedDerivative(_cache, _coeffs, 2, temp);
            temp *= _xd* t * dgdf;

            df = _expansion.MixedDerivative(_cache, _coeffs, 1, gradSeg);

            gradSeg *= ( _xd*t*df2*PosFuncType::SecondDerivative(df) + dgdf );
            gradSeg += temp;

        }else{
            df = _expansion.DiagonalDerivative(_cache, _coeffs, 1);
        }
        
        // First output is always the integrand itself
        double gf = PosFuncType::Evaluate(df);
        output(0) = _xd*gf;

        // Check for infs or nans
        if(std::isinf(gf)){
            std::stringstream msg;
            msg << "In MonotoneIntegrand, value of g(df(...)) is inf.  The value of df(...) is " << df << ", and the value of g(df(...)) is " << gf << ".";
            throw std::domain_error(msg.str());
        }else if(std::isnan(gf)){
            throw std::domain_error("In MonotoneIntegrand, A nan was encountered in value of g(df(...)).");
        }

        // Compute the derivative with respect to x_d
        if(_derivType==DerivativeFlags::Diagonal){
            
            // Compute \partial^2_d f
            output(1) = _expansion.DiagonalDerivative(_cache, _coeffs, 2);

            // Use the chain rule to get \partial_d g(f)
            output(1) *= _xd*t*PosFuncType::Derivative(df);
            output(1) += gf;
        }

        return output;
    }

private:

    const unsigned int _dim;
    double* _cache;
    ExpansionType const& _expansion;
    PointType const& _pt;
    double _xd;
    CoeffsType const& _coeffs;
    DerivativeFlags::DerivativeType _derivType;

}; // class MonotoneIntegrand


} // namespace mpart

#endif // #ifndef MPART_MONOTONEINTEGRAND_H