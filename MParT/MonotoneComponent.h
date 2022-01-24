#ifndef MPART_MONOTONECOMPONENT_H
#define MPART_MONOTONECOMPONENT_H

#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

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
    
    After the constructor has been called, cache[startPos[d]][p] will contain \phi_p(x_d)
 
   @tparam BasisEvaluatorType A class defining the family of 1d basis functions used to parameterize the function \f$f\f$.  The MParT::HermiteFunction and MParT::ProbabilistHermite classes are examples of types that implement the required interface.
   @tparam PosFuncType A class defining the function \f$g\f$.  This class must have `Evaluate` and `Derivative` functions accepting a double and returning a double.  The MParT::SoftPlus and MParT::Exp classes in PositiveBijectors.h are examples of classes defining this interface.
 */
template<class BasisEvaluatorType, class PosFuncType>
class CachedMonotoneIntegrand{
public:

    /**
      @param cache A pointer to memory storing evaluations of x_i^p for each i.  These terms need to be evaluated outside this class for \f$i\in\{0,\ldots,D-1\}\f$. 
      @param startPos Precomputed indices denoting the start of segments in cache for each dimension.
      @param maxDegree The maximum degree for each dimension.  Should be the same size as startPos.
      @param dim The length of both startPos and maxOrders.  If maxOrders was a vector, dim=maxOrders.size()

      @tparam ViewType2d A 2d Kokkos::View type
     */
    CachedMonotoneIntegrand(double*                            cache, 
                            Kokkos::View<unsigned int*> const& startPos,
                            Kokkos::View<const unsigned int*> const& maxDegrees,
                            FixedMultiIndexSet          const& multiSet,
                            Kokkos::View<double*>       const& coeffs) : _cache(cache),
                                                                         _startPos(startPos),
                                                                         _maxDegrees(maxDegrees),
                                                                         _dim(maxDegrees.extent(0)),
                                                                         _multiSet(multiSet),
                                                                         _coeffs(coeffs)
    {
    }

    /**
     Evaluates \f$g( \partial_d f(x_1,x_2,\ldots, t))\f$ using the cached values of \f$x\f$ given to the constructor 
     and the value of \f$t\f$ passed to this function.
    */
    double operator()(double t) const
    {   
        const unsigned int numTerms = _multiSet.Size();

        double output = 0;

        // Evaluate the orthgonal polynomial at the quadrature point
        _basis.EvaluateDerivatives(&_cache[_startPos(_dim-1)], _maxDegrees(_dim-1), t);
       
        // Compute coeff * polyval for each term
        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {   
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            bool hasDeriv = false;
            for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i){
                termVal *= _cache[_startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];
                if(_multiSet.nzDims(i)==_dim-1)
                    hasDeriv = true;
            }
            if(hasDeriv){
                // Multiply by the coefficients to get the contribution to the output
                output += termVal*_coeffs(termInd);
            }
        }

        return _posFunc.Evaluate(output);
    }

private:
    double* _cache;
    Kokkos::View<unsigned int*> const& _startPos;
    Kokkos::View<const unsigned int*> const& _maxDegrees;
    unsigned int _dim;

    BasisEvaluatorType _basis;
    PosFuncType _posFunc;
    FixedMultiIndexSet const& _multiSet;
    Kokkos::View<double*> const& _coeffs;
};


/**
Defines a function \f$T:R^N\rightarrow R\f$ such that \f$\partial T / \partial x_N >0\f$ is strictly positive.

The function \f$T\f$ is based on another (generally non-monotone) function \f$f : R^N\rightarrow R\f$ and a strictly positve function 
\f$g : R\rightarrow R^+\f$.   Together, these functions define the monotone component $T$ through

$$
T(x_1, x_2, ..., x_N) = f(x_1,x_2,..., x_{N-1}, 0) + \int_0^{x_N}  g\left( \frac{\partial f}{\partial x_d}(x_1,x_2,..., x_{N-1}, t) \right) dt
$$

@tparam BasisEvaluatorType A class defining the family of 1d basis functions used to parameterize the function \f$f\f$.  The MParT::HermiteFunction and MParT::ProbabilistHermite classes are examples of types that implement the required interface.
@tparam PosFuncType A class defining the function \f$g\f$.  This class must have `Evaluate` and `Derivative` functions accepting a double and returning a double.  The MParT::SoftPlus and MParT::Exp classes in PositiveBijectors.h are examples of classes defining this interface.
@tparam QuadratureType A class defining the integration scheme used to approximate \f$\int_0^{x_N}  g\left( \frac{\partial f}{\partial x_d}(x_1,x_2,..., x_{N-1}, t) \right) dt\f$.  The type must have a function `Integrate(f,lb,ub)` that accepts a functor `f`, a double lower bound `lb`, a double upper bound `ub`, and returns a double with an estimate of the integral.   The MParT::AdaptiveSimpson and MParT::RecursiveQuadrature classes provide this interface. 
*/
template<class BasisEvaluatorType, class PosFuncType, class QuadratureType>
class MonotoneComponent
{

public:

    MonotoneComponent(MultiIndexSet const& multiSet, 
                      QuadratureType     const& quad) : MonotoneComponent(multiSet.Fix(), quad){}

    MonotoneComponent(FixedMultiIndexSet const& multiSet, 
                      QuadratureType     const& quad) : _multiSet(multiSet), _quad(quad)
    {
    }

    /**
       @brief Evaluates the monotone function \f$T(x_1,\ldots,x_D)\f$ at multiple points.
       
     * @param pts A \f$D\times N\f$ array containing the \f$N\f$ points in \f$\mathbb{R}^D\f$ where we want to evaluate the monotone component.  Each column is a point.
     * @param coeffs The coefficients in the expansion defining \f$f\f$.  The length of this array must be the same as the number of terms in the multiindex set passed to the constructor.
     * @return Kokkos::View<double*> An array containing the evaluattions \f$T(x^{(i)}_1,\ldots,x^{(i)}_D)\f$ for each \f$i\in\{0,\ldots,N}\f$.
     */
    Kokkos::View<double*> Evaluate(Kokkos::View<double**> const& pts, 
                                   Kokkos::View<double*> const& coeffs)
    {   
        unsigned int numPts = pts.extent(1);
        unsigned int numTerms = _multiSet.Size();
        unsigned int dim = pts.extent(0);

        assert(coeffs.extent(0)==numTerms);

        Kokkos::View<double*> output("ExpansionOutput", numPts);

        // Figure how much scratch space is needed to store the cached values 
        Kokkos::View<const unsigned int*> maxDegrees = _multiSet.MaxDegrees();
        Kokkos::View<unsigned int*> startPos("Indices for start of 1d basis evaluations", maxDegrees.extent(0)+1);
        startPos(0) = 0;
        for(unsigned int i=1; i<maxDegrees.extent(0)+1; ++i)
            startPos(i) = startPos(i-1) + maxDegrees(i-1)+1;

        unsigned int cacheSize = startPos[maxDegrees.extent(0)];

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));


        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            unsigned int ptInd = team_member.league_rank();

            // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
            BasisEvaluatorType basis;
            double* polyCache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            for(unsigned int d=0; d<dim-1; ++d)
                basis.EvaluateAll(&polyCache[startPos(d)], maxDegrees(d), pts(d,ptInd));
            
            // Create the integrand
            CachedMonotoneIntegrand<BasisEvaluatorType, PosFuncType> integrand(polyCache, startPos, maxDegrees, _multiSet, coeffs);
            output(ptInd) = _quad.Integrate(integrand, 0, pts(dim-1,ptInd));

            // Now add the f(x_1,...,x_{D-1},0) evaluation that lies outside the integral
            basis.EvaluateAll(&polyCache[startPos(dim-1)], maxDegrees(dim-1), 0.0);

            for(unsigned int termInd=0; termInd<numTerms; ++termInd)
            {   
                // Compute the value of this term in the expansion
                double termVal = 1.0;
                for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i)
                    termVal *= polyCache[startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];

                // Multiply by the coefficients to get the contribution to the output
                output(ptInd) += termVal*coeffs(termInd);
            }
            
        });

        return output;
    }

private:
    FixedMultiIndexSet _multiSet;
    QuadratureType _quad;
};

} // namespace mpart
#endif 