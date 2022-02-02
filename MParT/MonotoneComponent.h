#ifndef MPART_MONOTONECOMPONENT_H
#define MPART_MONOTONECOMPONENT_H

#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>

namespace mpart{

enum DerivativeType {
        None,
        Parameters,
        Diagonal
    };

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
      @param cache A pointer to memory storing evaluations of \phi_{i,p}(x_i) for each i.  These terms need to be evaluated outside this class for \f$i\in\{0,\ldots,D-1\}\f$. 
      @param startPos Precomputed indices denoting the start of segments in cache for each dimension.
      @param maxDegree The maximum degree for each dimension.  Should be the same size as startPos.
      @param dim The length of both startPos and maxOrders.  If maxOrders was a vector, dim=maxOrders.size()
      @param computeDeriv Whether or not the derviative \f$\partial T_d / \partial x_d\f$ should be computed as well.  If true, the operator in this function will return a two component vector.  The first component is the integrand necessary to evaluate the function \f$T_d\f$ itself and the second is the integrand needed to compute \f$\partial T_d / \partial x_d\f$.

      @tparam ViewType2d A 2d Kokkos::View type
     */
    CachedMonotoneIntegrand(double*                            cache,
                            double                             xd,
                            Kokkos::View<unsigned int*> const& startPos,
                            Kokkos::View<const unsigned int*> const& maxDegrees,
                            FixedMultiIndexSet          const& multiSet,
                            Kokkos::View<double*>       const& coeffs,
                            DerivativeType                     derivType) : _cache(cache),
                                                                               _xd(xd),
                                                                               _startPos(startPos),
                                                                               _maxDegrees(maxDegrees),
                                                                               _dim(maxDegrees.extent(0)),
                                                                               _multiSet(multiSet),
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
        const unsigned int numTerms = _multiSet.Size();

        unsigned int numOutputs = 1;
        if(_derivType==DerivativeType::Diagonal)
            numOutputs++;
        if(_derivType==DerivativeType::Parameters)
            numOutputs += numTerms;

        Eigen::VectorXd output = Eigen::VectorXd::Zero(numOutputs);

        double f = 0;

        // Evaluate the orthogonal polynomial and its derivatives at the quadrature point
        if(_derivType==DerivativeType::Diagonal){
            _basis.EvaluateSecondDerivatives(&_cache[_startPos(_dim-1)], // basis vals
                                             &_cache[_startPos(_dim)],   // basis derivatives
                                             &_cache[_startPos(_dim+1)], // basis second derivatives 
                                              _maxDegrees(_dim-1),       // largest basis degree
                                              t*_xd);                    // point to evaluate at
        }else{
            _basis.EvaluateDerivatives(&_cache[_startPos(_dim-1)], // basis vals
                                       &_cache[_startPos(_dim)], // basis derivatives
                                        _maxDegrees(_dim-1),       // largest basis degree
                                        t*_xd);                    // point to evaluate at
        }
       
        // Compute coeff * polyval for each term
        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {   
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            bool hasDeriv = false;
            for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i){
                if(_multiSet.nzDims(i)==_dim-1){
                    termVal *= _cache[_startPos(_dim) + _multiSet.nzOrders(i)];
                    hasDeriv = true;
                }else{
                    termVal *= _cache[_startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];
                }
                
            }
            if(hasDeriv){
                // Multiply by the coefficients to get the contribution to the output
                f += termVal*_coeffs(termInd);

                if(_derivType==DerivativeType::Parameters)
                    output(termInd+1) = termVal;
            }
        }

        double gf = PosFuncType::Evaluate(f);

        if(_derivType==DerivativeType::Parameters)
            output *= _xd*PosFuncType::Derivative(f);

        output(0) = _xd*gf;
        
        // Compute the derivative with respect to x_d
        if(_derivType==DerivativeType::Diagonal){
            
            double deriv = 0;
            
            // Compute coeff * polyval for each term
            for(unsigned int termInd=0; termInd<numTerms; ++termInd)
            {   
                // Compute the value of this term in the expansion
                double termVal = 1.0;
                bool hasDeriv = false;
                for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i){
                    
                    if(_multiSet.nzDims(i)==_dim-1){
                        termVal *= _cache[_startPos(_dim+1) + _multiSet.nzOrders(i)];
                        hasDeriv = true;
                    }else{
                        termVal *= _cache[_startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];
                    }
                }
                if(hasDeriv){
                    // Multiply by the coefficients to get the contribution to the output
                    deriv += termVal*_coeffs(termInd);
                }
            }

            // The deriv variable currently holds the second derivative of f

            deriv *= _xd*t*PosFuncType::Derivative(f);
            deriv += gf;

            output(1) = deriv;
        }

        return output;
    }

private:
    double* _cache;

    double  _xd;
    Kokkos::View<unsigned int*> const& _startPos;
    Kokkos::View<const unsigned int*> const& _maxDegrees;
    unsigned int _dim;

    BasisEvaluatorType _basis;
    FixedMultiIndexSet const& _multiSet;
    Kokkos::View<double*> const& _coeffs;

    DerivativeType _derivType;
};


/**
@brief Defines a function \f$T:R^N\rightarrow R\f$ such that \f$\partial T / \partial x_N >0\f$ is strictly positive.

@details 
The function \f$T\f$ is based on another (generally non-monotone) function \f$f : R^N\rightarrow R\f$ and a strictly positve function 
\f$g : R\rightarrow R_{>0}\f$.   Together, these functions define the monotone component $T$ through

$$
T(x_1, x_2, ..., x_D) = f(x_1,x_2,..., x_{D-1}, 0) + \int_0^{x_D}  g\left( \partial_D f(x_1,x_2,..., x_{D-1}, t) \right) dt
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
     * @return Kokkos::View<double*> An array containing the evaluattions \f$T(x^{(i)}_1,\ldots,x^{(i)}_D)\f$ for each \f$i\in\{0,\ldots,N\}\f$.
     */
    Kokkos::View<double*> Evaluate(Kokkos::View<double**> const& pts, 
                                   Kokkos::View<double*> const& coeffs)
    {   
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = _multiSet.Size();
        const unsigned int dim = pts.extent(0);

        assert(coeffs.extent(0)==numTerms);

        Kokkos::View<double*> output("ExpansionOutput", numPts);

        // Figure how much scratch space is needed to store the cached values 
        Kokkos::View<const unsigned int*> maxDegrees = _multiSet.MaxDegrees();
        Kokkos::View<unsigned int*> startPos("Indices for start of 1d basis evaluations", dim+2); // each dimension + one for derivatives + one for end
       
        startPos(0) = 0;
        for(unsigned int i=1; i<dim+1; ++i)
            startPos(i) = startPos(i-1) + maxDegrees(i-1)+1;
        startPos(dim+1) = startPos(dim) + maxDegrees(dim-1)+1;
        
        unsigned int cacheSize = startPos(dim+1);

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));


        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            unsigned int ptInd = team_member.league_rank();

            // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
            BasisEvaluatorType basis;
            double* polyCache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            // Evaluate all degrees of all 1d polynomials except the last dimension, which will be evaluated inside the integrand
            for(unsigned int d=0; d<dim-1; ++d)
                basis.EvaluateAll(&polyCache[startPos(d)], maxDegrees(d), pts(d,ptInd));

            // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
            CachedMonotoneIntegrand<BasisEvaluatorType, PosFuncType> integrand(polyCache, pts(dim-1,ptInd), startPos, maxDegrees, _multiSet, coeffs, DerivativeType::None);
            
            // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt
            output(ptInd) = _quad.Integrate(integrand, 0, 1)(0);

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

    /**
       @brief Approximates the "continuous derivative" \f$\frac{\partial T}{\partial x_D}\f$ derived from the exact integral form of the transport map.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param coeffs The ceofficients in an expansion for \f$f\f$.
        @return Kokkos::View<double*> The values of \f$\frac{\partial T}{\partial x_D}\f$ at each point.
     */
    Kokkos::View<double*>  ContinuousDerivative(Kokkos::View<double**> const& pts, 
                                              Kokkos::View<double*>  const& coeffs)
    {   
        const unsigned int numPts = pts.extent(1);
        Kokkos::View<double*> derivs("Component Derivatives", numPts);
        
        ContinuousDerivative(pts,coeffs, derivs);

        return derivs;
    }

    /**
        @brief Approximates the "continuous derivative" \f$\frac{\partial T}{\partial x_D}\f$ derived from the exact integral form of the transport map.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives.
        @param[in] pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param[in] coeffs The ceofficients in an expansion for \f$f\f$.
        @param[in,out] evals  The values of map component itself \f$T\f$ at each point.
        @param[in,out] derivs The values of \f$ \frac{\partial T}{\partial x_D}\f$ at each point.
     */
    void ContinuousDerivative(Kokkos::View<double**> const& pts, 
                              Kokkos::View<double*>  const& coeffs,
                              Kokkos::View<double*>       & derivs)
    {   
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = _multiSet.Size();
        const unsigned int dim = pts.extent(0);

        assert(coeffs.extent(0)==numTerms);

        // Figure how much scratch space is needed to store the cached values 
        Kokkos::View<const unsigned int*> maxDegrees = _multiSet.MaxDegrees();
        Kokkos::View<unsigned int*> startPos("Indices for start of 1d basis evaluations", dim+2); // each dimension + one for derivatives + one for end
        
        startPos(0) = 0;
        for(unsigned int i=1; i<dim+1; ++i)
            startPos(i) = startPos(i-1) + maxDegrees(i-1)+1;
        startPos(dim+1) = startPos(dim) + maxDegrees(dim-1)+1;

        unsigned int cacheSize = startPos(dim+1);

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));

        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            unsigned int ptInd = team_member.league_rank();

            // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
            BasisEvaluatorType basis;
            double* polyCache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            // Precompute all off-diagonal 1d values
            for(unsigned int d=0; d<dim-1; ++d)
                basis.EvaluateAll(&polyCache[startPos(d)], maxDegrees(d), pts(d,ptInd));
            
            // Evaluate df/dx_d(x_1,...,x_d)
            basis.EvaluateDerivatives(&polyCache[startPos(dim-1)], // Holds the values
                                      &polyCache[startPos(dim)],  // Holds the derivatives 
                                       maxDegrees(dim-1), 
                                       pts(dim-1,ptInd));
            derivs(ptInd) = 0.0;

            for(unsigned int termInd=0; termInd<numTerms; ++termInd)
            {   
                // Compute the value of this term in the expansion
                double termVal = 1.0;
                bool hasDeriv = false;
                for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i){
                    if(_multiSet.nzDims(i)==dim-1){
                        termVal *= polyCache[startPos(dim) + _multiSet.nzOrders(i)];
                        hasDeriv = true;
                    }else{
                        termVal *= polyCache[startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];
                    }
                }

                // Multiply by the coefficients to get the contribution to the output
                if(hasDeriv)
                    derivs(ptInd) += termVal*coeffs(termInd);
            }

            // Compute g(df/dx)
            derivs(ptInd) = PosFuncType::Evaluate(derivs(ptInd));
        });
    }

    
    Kokkos::View<double*>  DiscreteDerivative(Kokkos::View<double**> const& pts, 
                                              Kokkos::View<double*>  const& coeffs)
    {   
        const unsigned int numPts = pts.extent(1);
        Kokkos::View<double*> evals("Component Evaluations", numPts);
        Kokkos::View<double*> derivs("Component Derivatives", numPts);
        
        DiscreteDerivative(pts,coeffs, evals, derivs);

        return derivs;
    }

    /**
     * @brief Approximates the "discrete derivative" of the quadrature-based approximation \f$\tilde{T}\f$.
        @details See the <a href="../getting_started/mathematics.html">mathematical background</a> section for more details on discrete and continuous map derivatives. `mathematics`_
        @param[in] pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param[in] coeffs The ceofficients in an expansion for \f$f\f$.
        @param[in,out] evals  The values of map component itself \f$\tilde{T}\f$ at each point.
        @param[in,out] derivs Kokkos::View<double*> The values of \f$ \frac{\partial \tilde{T}}{\partial x_D}\f$ at each point.
     */
    void  DiscreteDerivative(Kokkos::View<double**> const& pts, 
                             Kokkos::View<double*>  const& coeffs,
                             Kokkos::View<double*>       & evals, 
                             Kokkos::View<double*>       & derivs)
    {   
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = _multiSet.Size();
        const unsigned int dim = pts.extent(0);

        assert(coeffs.extent(0)==numTerms);

        Kokkos::View<double*> output("ExpansionOutput", numPts);

        // Figure how much scratch space is needed to store the cached values 
        Kokkos::View<const unsigned int*> maxDegrees = _multiSet.MaxDegrees();

        Kokkos::View<unsigned int*> startPos("Indices for start of 1d basis evaluations", maxDegrees.extent(0)+3); // each dimension + derivs in d + 2nd derivs in d + ending
        startPos(0) = 0;
        for(unsigned int i=1; i<dim+1; ++i)
            startPos(i) = startPos(i-1) + maxDegrees(i-1)+1;
        startPos(dim+1) = startPos(dim) + maxDegrees(dim-1)+1;
        startPos(dim+2) = startPos(dim+1) + maxDegrees(dim-1)+1;
        
        
        unsigned int cacheSize = startPos(dim+2);

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));


        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            unsigned int ptInd = team_member.league_rank();

            // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
            BasisEvaluatorType basis;
            double* polyCache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            // Evaluate all of the 1d orthogonal polynomials except the last dimension
            for(unsigned int d=0; d<dim-1; ++d)
                basis.EvaluateAll(&polyCache[startPos(d)], maxDegrees(d), pts(d,ptInd));
            
            
            // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
            CachedMonotoneIntegrand<BasisEvaluatorType, PosFuncType> integrand(polyCache, pts(dim-1,ptInd), startPos, maxDegrees, _multiSet, coeffs, DerivativeType::Diagonal);
            
            // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt
            Eigen::VectorXd both = _quad.Integrate(integrand, 0, 1);

            evals(ptInd) = both(0);
            derivs(ptInd) = both(1);

            // Now add the f(x_1,...,x_{D-1},0) evaluation that lies outside the integral
            basis.EvaluateAll(&polyCache[startPos(dim-1)], maxDegrees(dim-1), 0.0);

            for(unsigned int termInd=0; termInd<numTerms; ++termInd)
            {   
                // Compute the value of this term in the expansion
                double termVal = 1.0;
                for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i)
                    termVal *= polyCache[startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];

                // Multiply by the coefficients to get the contribution to the output
                evals(ptInd) += termVal*coeffs(termInd);
            }
            
        });
    }

    /** @brief Returns the gradient of the map with respect to the parameters \f$\mathbf{w}\f$.

        @details 
        Consider \f$N\f$ points \f$\{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(N)}\}\f$ and let 
        \f$y_d^{(i)} = T_d(\mathbf{x}^{(i)}; \mathbf{w})\f$.   This function returns the gradient 
        \f$\nabla_{\mathbf{w}} J\f$ for some objective function \f$J(y_d^{(1)}, \ldots, y_d^{(N)})\f$
        depending on the map component outputs.   The vector \f$\mathbf{s}\in\mathbb{R}^N\f$ 
        contains the sensitivities \f$\mathbf{s} = \left[\frac{\partial J}{\partial y_d^{(1)}}, \ldots, \frac{\partial J}{\partial y_d^{(N)}} \right]\f$ 
        of the objective \f$J\f$ with respect to each \f$y_d^{(i)}\f$.
        
        @param[in] pts A \f$D\times N\f$ matrix containing the points \f$x^{(1)},\ldots,x^{(N)}\f$.  Each column is a point.
        @param[in] coeffs A vector of coefficients defining the function \f$f(\mathbf{x}; \mathbf{w})\f$.
        @param[in] sens A vector of sensitivities.
        @returns The gradient \f$\nabla_{\mathbf{w}} J\f$.  This is equivalent to applying the transpose of the Jacobian \f$\nabla_{\mathbf{w}} T_d\f$ to \f$\mathbf{s}\f$.
    */
    Kokkos::View<double*> CoeffGradient(Kokkos::View<double**> const& pts, 
                                        Kokkos::View<double*>  const& coeffs,
                                        Kokkos::View<double*>  const& sens)
    {
        const unsigned int numPts = pts.extent(1);
        const unsigned int numTerms = _multiSet.Size();
        const unsigned int dim = pts.extent(0);

        assert(coeffs.extent(0)==numTerms);

        Kokkos::View<double*> gradient("Coefficient Gradient", numTerms);

        // Figure how much scratch space is needed to store the cached values 
        Kokkos::View<const unsigned int*> maxDegrees = _multiSet.MaxDegrees();
        Kokkos::View<unsigned int*> startPos("Indices for start of 1d basis evaluations", dim+2); // each dimension + one for derivatives + one for end
       
        startPos(0) = 0;
        for(unsigned int i=1; i<dim+1; ++i)
            startPos(i) = startPos(i-1) + maxDegrees(i-1)+1;
        startPos(dim+1) = startPos(dim) + maxDegrees(dim-1)+1;
        
        unsigned int cacheSize = startPos(dim+1);

        // Create a policy with enough scratch memory to cache the polynomial evaluations
        Kokkos::TeamPolicy<> policy = Kokkos::TeamPolicy<>(numPts, 1).set_scratch_size(1,Kokkos::PerTeam(cacheSize*sizeof(double)), Kokkos::PerThread(0));


        // Paralel loop over each point computing T(x_1,...,x_D) for that point
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA (auto team_member) {
            
            unsigned int ptInd = team_member.league_rank();

            // Evaluate the orthgonal polynomials in each direction (except the last) for all possible orders
            BasisEvaluatorType basis;
            double* polyCache = (double*) team_member.team_shmem().get_shmem(cacheSize*sizeof(double));

            // Evaluate all degrees of all 1d polynomials except the last dimension, which will be evaluated inside the integrand
            for(unsigned int d=0; d<dim-1; ++d)
                basis.EvaluateAll(&polyCache[startPos(d)], maxDegrees(d), pts(d,ptInd));

            // Create the integrand g( \partial_D f(x_1,...,x_{D-1},t))
            CachedMonotoneIntegrand<BasisEvaluatorType, PosFuncType> integrand(polyCache, pts(dim-1,ptInd), startPos, maxDegrees, _multiSet, coeffs, DerivativeType::Parameters);
            
            // Compute \int_0^x g( \partial_D f(x_1,...,x_{D-1},t)) dt
            Eigen::VectorXd integral = _quad.Integrate(integrand, 0, 1);
            
            // Now add the f(x_1,...,x_{D-1},0) evaluation that lies outside the integral
            basis.EvaluateAll(&polyCache[startPos(dim-1)], maxDegrees(dim-1), 0.0);

            for(unsigned int termInd=0; termInd<numTerms; ++termInd)
            {      
                // Compute the value of this term in the expansion
                double termVal = 1.0;
                for(unsigned int i=_multiSet.nzStarts(termInd); i<_multiSet.nzStarts(termInd+1); ++i)
                    termVal *= polyCache[startPos(_multiSet.nzDims(i)) + _multiSet.nzOrders(i)];

                // Multiply by the coefficients to get the contribution to the output
                Kokkos::atomic_add(&gradient(termInd), sens(ptInd) * (termVal + integral(termInd+1)));
            }
            
        });

        return gradient;
    }


    /** The map \f$T(x_1, x_2, ..., x_D; w)\f$ is parameterized by coefficients \f$w\f$.  This function computes
        mixed second derivatives \f$ \nabla_w \left[\frac{\partial}{\partial x_D} T(x_1, x_2, ..., x_D; w)\right]\f$ 
        at multiple points.  Columns in the output correspond to points.  Rows correspond to input dimensions.

        @param pts  The points where we want to evaluate the derivative.  Each column represents one point.
        @param coeffs The ceofficients in an expansion characterizing \f$f\f$.
        @return Kokkos::View<double**> The values of \f$\nabla_w \left[\frac{\partial}{\partial x_D} T(x_1, x_2, ..., x_D; w)\f$ at each point.
    */
    Kokkos::View<double**> MixedSecondDerivative(Kokkos::View<double**> const& pts, 
                                                 Kokkos::View<double*>  const& coeffs)
    {   
        
    }

private:
    FixedMultiIndexSet _multiSet;
    QuadratureType _quad;
};

} // namespace mpart
#endif 