#ifndef MPART_MONOTONEINTEGRAND_H
#define MPART_MONOTONEINTEGRAND_H

#include "MParT/DerivativeFlags.h"

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
template<class ExpansionType, class PosFuncType, class PointType, class CoeffsType, typename MemorySpace=Kokkos::HostSpace>
class MonotoneIntegrand{
public:



    /**
      @param cache A pointer to memory storing evaluations of \phi_{i,p}(x_i) for each i.  These terms need
                   to be evaluated outside this class (e.g., using `expansion_.FillCache1` for \f$i\in\{0,\ldots,D-1\}\f$.
      @param expansion
      @param xd
      @param coeffs
      @param derivType
     */
    KOKKOS_INLINE_FUNCTION MonotoneIntegrand(double*                            cache,
                                             ExpansionType               const& expansion,
                                             PointType                   const& pt,
                                             CoeffsType                  const& coeffs,
                                             DerivativeFlags::DerivativeType    derivType) : MonotoneIntegrand(cache, expansion, pt, pt(pt.extent(0)-1), coeffs, derivType)
    {
    }

    KOKKOS_INLINE_FUNCTION MonotoneIntegrand(double*                            cache,
                                             ExpansionType               const& expansion,
                                             PointType                   const& pt,
                                             double                             xd,
                                             CoeffsType                  const& coeffs,
                                             DerivativeFlags::DerivativeType    derivType) : dim_(pt.extent(0)),
                                                                                            cache_(cache),
                                                                                            expansion_(expansion),
                                                                                            pt_(pt),
                                                                                            xd_(xd),
                                                                                            coeffs_(coeffs),
                                                                                            derivType_(derivType)
    {
        assert(derivType!=DerivativeFlags::Mixed);
        assert(derivType!=DerivativeFlags::MixedInput);
    }

    KOKKOS_INLINE_FUNCTION MonotoneIntegrand(double*                            cache,
                                             ExpansionType               const& expansion,
                                             PointType                   const& pt,
                                             CoeffsType                  const& coeffs,
                                             DerivativeFlags::DerivativeType    derivType,
                                             Kokkos::View<double*, MemorySpace> workspace) : MonotoneIntegrand(cache, expansion, pt, pt(pt.extent(0)-1), coeffs, derivType, workspace)
    {
    }

    KOKKOS_INLINE_FUNCTION MonotoneIntegrand(double*                            cache,
                                             ExpansionType               const& expansion,
                                             PointType                   const& pt,
                                             double                             xd,
                                             CoeffsType                  const& coeffs,
                                             DerivativeFlags::DerivativeType    derivType,
                                             Kokkos::View<double*, MemorySpace> workspace) : dim_(pt.extent(0)),
                                                                                            cache_(cache),
                                                                                            expansion_(expansion),
                                                                                            pt_(pt),
                                                                                            xd_(xd),
                                                                                            coeffs_(coeffs),
                                                                                            derivType_(derivType),
                                                                                            _workspace(workspace)
    {
        if(derivType==DerivativeFlags::Mixed)
            assert(workspace.extent(0)>=coeffs.extent(0));
    }



    /**
     Evaluates \f$g( \partial_d f(x_1,x_2,\ldots, x_d*t))\f$ using the cached values of \f$x\f$ given to the constructor
     and the value of \f$t\f$ passed to this function.  Note that we assume t ranges from [0,1].  The change of variables to x_d*t is
     taken care of inside this function.
    */
    KOKKOS_INLINE_FUNCTION void operator()(double t, double* output) const
    {
        const unsigned int numTerms = expansion_.NumCoeffs();
        const unsigned int dim = pt_.size();

        unsigned int numOutputs = 1;
        if(derivType_==DerivativeFlags::Diagonal)
            numOutputs++;
        if((derivType_==DerivativeFlags::Parameters) || (derivType_==DerivativeFlags::Mixed))
            numOutputs += numTerms;
        if((derivType_==DerivativeFlags::Input) || (derivType_==DerivativeFlags::MixedInput))
            numOutputs += dim;

        // Finish filling in the cache at the quadrature point (FillCache1 is called outside this class)
        if((derivType_==DerivativeFlags::Diagonal)||(derivType_==DerivativeFlags::Mixed)||(derivType_==DerivativeFlags::Input)){
            expansion_.FillCache2(cache_, pt_, t*xd_, DerivativeFlags::Diagonal2);
        }else{
            expansion_.FillCache2(cache_, pt_, t*xd_, DerivativeFlags::Diagonal);
        }

        // Use the cache to evaluate \partial_d f and, optionally, the gradient of \partial_d f wrt the coefficients or input.
        double df = 0;
        if(derivType_==DerivativeFlags::Parameters){
            Kokkos::View<double*,MemorySpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> gradSeg(&output[1], numTerms);
            df = expansion_.MixedCoeffDerivative(cache_, coeffs_, 1, gradSeg);

            double scale = xd_*PosFuncType::Derivative(df);
            for(unsigned int i=0; i<numTerms;++i)
                gradSeg(i) *= scale;

        }else if(derivType_==DerivativeFlags::Mixed){

            df = expansion_.DiagonalDerivative(cache_, coeffs_, 1);

            double dgdf = PosFuncType::Derivative(df);
            double df2 = expansion_.MixedCoeffDerivative(cache_, coeffs_, 2, _workspace);

            double scale = xd_* t * dgdf;
            for(unsigned int i=0; i<numTerms; ++i)
                _workspace(i) *= scale;

            Kokkos::View<double*,MemorySpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> gradSeg(&output[1], numTerms);
            df = expansion_.MixedCoeffDerivative(cache_, coeffs_, 1, gradSeg);

            scale = xd_*t*df2*PosFuncType::SecondDerivative(df) + dgdf;
            for(unsigned int i=0; i<numTerms; ++i)
                gradSeg(i) = scale*gradSeg(i) + _workspace(i);

        }else if(derivType_==DerivativeFlags::Input){
            
            Kokkos::View<double*,MemorySpace,Kokkos::MemoryTraits<Kokkos::Unmanaged>> gradSeg(&output[1], dim);
            df = expansion_.MixedInputDerivative(cache_, coeffs_, gradSeg);

            double scale = xd_*PosFuncType::Derivative(df);
            for(unsigned int i=0; i<dim-1;++i)
                gradSeg(i) *= scale;

        }else{
            df = expansion_.DiagonalDerivative(cache_, coeffs_, 1);
        }

        // First output is always the integrand itself
        double gf = PosFuncType::Evaluate(df);
        output[0] = xd_*gf;
        
        // Check for infs or nans
        if(std::isinf(gf)){
            printf("\nERROR: In MonotoneIntegrand, value of g(df(...)) is inf.  The value of df(...) is %0.4f, and the value of f(df(...)) is %0.4f.\n\n", df, gf);
        }else if(std::isnan(gf)){
            printf("\nERROR: In MonotoneIntegrand, A nan was encountered in value of g(df(...)).\n\n");
        }

        // Compute the derivative with respect to x_d
        if((derivType_==DerivativeFlags::Diagonal) || (derivType_==DerivativeFlags::Input)){
            
            unsigned int ind = (derivType_==DerivativeFlags::Diagonal) ? 1 : dim;
            // Compute \partial^2_d f
            output[ind] = expansion_.DiagonalDerivative(cache_, coeffs_, 2);

            // Use the chain rule to get \partial_d g(f)
            output[ind] *= xd_*t*PosFuncType::Derivative(df);
            output[ind] += gf;
        }
    }

private:

    const unsigned int dim_;
    double* cache_;
    ExpansionType const& expansion_;
    PointType const& pt_;
    double xd_;
    CoeffsType const& coeffs_;
    DerivativeFlags::DerivativeType derivType_;
    Kokkos::View<double*,MemorySpace> _workspace;

}; // class MonotoneIntegrand


} // namespace mpart

#endif // #ifndef MPART_MONOTONEINTEGRAND_H