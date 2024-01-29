#ifndef MPART_MultivariateExpansionWorker_H
#define MPART_MultivariateExpansionWorker_H

#include <Kokkos_Core.hpp>

#include "MParT/DerivativeFlags.h"

#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/BasisEvaluator.h"


namespace mpart{

template<typename MemorySpace>
struct MultivariateExpansionMaxDegreeFunctor {

    MultivariateExpansionMaxDegreeFunctor(unsigned int dim, Kokkos::View<unsigned int*, MemorySpace> startPos, Kokkos::View<const unsigned int*, MemorySpace> maxDegreesIn) : dim(dim), startPos(startPos), maxDegrees(maxDegreesIn)
    {
    };

    KOKKOS_FUNCTION void operator()(const unsigned int i, unsigned int& update, const bool final) const{
        if(final)
            startPos(i) = update;

        if(i<2*dim){
            update += maxDegrees(i % dim)+1;
        }else{
            update += maxDegrees(dim-1)+1;
        }
    };

    unsigned int dim;
    Kokkos::View<unsigned int*, MemorySpace> startPos;
    Kokkos::View<const unsigned int*, MemorySpace> maxDegrees;
};

template<typename MemorySpace>
struct CacheSizeFunctor{

    CacheSizeFunctor(Kokkos::View<unsigned int*, MemorySpace> startPosIn, Kokkos::View<unsigned int*, MemorySpace> cacheSizeIn) : startPos_(startPosIn), cacheSize_(cacheSizeIn){};

    KOKKOS_INLINE_FUNCTION void operator()(const int) const{cacheSize_(0) = startPos_(startPos_.extent(0)-1);};

    Kokkos::View<unsigned int*, MemorySpace> startPos_;
    Kokkos::View<unsigned int*, MemorySpace> cacheSize_;

};

/**
 @brief Defines a function in terms of the tensor product of unary basis functions.
 @details
 - Cache memory managed elsewhere
 \f\[
     \text{cache} = \left[\begin{array}{c}
     \phi_1^0(x_1)\\
     \phi_1^1(x_1)\\
     \vdots\\
     \phi_1^{p_1}\\
     \phi_2^0(x_2)\\
     \vdots\\
     \phi_2^{p_2}(x_2)\\
     \vdots \\
     \phi_d^{p_d}(x_d)\\
     \frac{\partial}{\partial x_d}\phi_d^0(x_d)\\
     \vdots\\
     \frac{\partial}{\partial x_d}\phi_d^{p_d}(x_d)\\
     \frac{\partial^2}{\partial x_d^2}\phi_d^0(x_d^2)\\
     \vdots\\
     \frac{\partial^2}{\partial x_d^2}\phi_d^{p_d}(x_d)
     \end{array}
     \right]
  \f\]

 @tparam BasisEvaluatorType The family of 1d basis functions to employ.
 */
template<class BasisEvaluatorType, typename MemorySpace=Kokkos::HostSpace>
class MultivariateExpansionWorker
{
public:
    using BasisType = BasisEvaluatorType;
    using KokkosSpace = MemorySpace;
    using Rectifier = typename GetRectifier<BasisEvaluatorType>::type;

    MultivariateExpansionWorker() : dim_(0), multiSet_(FixedMultiIndexSet<MemorySpace>(1,0)){};

    MultivariateExpansionWorker(MultiIndexSet const& multiSet,
                                BasisEvaluatorType const& basis1d = BasisEvaluatorType()) : MultivariateExpansionWorker(multiSet.Fix(), basis1d){};

    MultivariateExpansionWorker(FixedMultiIndexSet<MemorySpace> const& multiSet,
                                BasisEvaluatorType const& basis1d = BasisEvaluatorType()) : dim_(multiSet.Length()),
                                                                                      multiSet_(multiSet),
                                                                                      basis1d_(basis1d),
                                                                                      startPos_("Indices for start of 1d basis evaluations", 2*multiSet.Length()+2),
                                                                                      maxDegrees_(multiSet_.MaxDegrees())
    {
        Kokkos::parallel_scan(Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,2*dim_+2), MultivariateExpansionMaxDegreeFunctor<MemorySpace>(dim_,startPos_, maxDegrees_));

        // Compute the cache size and store in a double on the host
        Kokkos::View<unsigned int*, MemorySpace> dCacheSize("Temporary cache size",1);
        Kokkos::parallel_for(Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,1), CacheSizeFunctor<MemorySpace>(startPos_, dCacheSize));
        cacheSize_ = ToHost(dCacheSize)(0);
    };

    /**
     @brief Returns the size of the cache needed to evaluate the expansion (in terms of number of doubles).
     @return unsigned int  The length of the required cache vector.
     */
    KOKKOS_INLINE_FUNCTION unsigned int CacheSize() const {
        return cacheSize_;
        //return startPos_(startPos_.extent(0)-1);
    };

    /**
     @brief Returns the number of coefficients in this expansion.
     @return unsigned int The number of terms in the multiindexset, which corresponds to the number of coefficients needed to define the expansion.
     */
    KOKKOS_INLINE_FUNCTION unsigned int NumCoeffs() const {return multiSet_.Size();};

    /**
    @brief Returns the dimension of inputs to this multivariate expansion.
    @return unsigned int The dimension of an input point.
    */
    KOKKOS_INLINE_FUNCTION unsigned int InputSize() const {return multiSet_.Length();};

    /**
     @brief Precomputes parts of the cache using all but the last component of the point, i.e., using only \f$x_1,x_2,\ldots,x_{d-1}\f$, not \f$x_d\f$.
     @details
     @tparam PointType The vector type used to define the point.  Can be anything allowing access to components with operator().  Examples are Kokkos::View<double*> or Eigen::VectorXd.  Only the first d-1 components of the vector will be accessed in this function.
     @param polyCache A pointer to the start of the cache.  This memory must be allocated before calling this function.
     @param pt The point (at least the first \f$d-1\f$ components) to use when filling in the cache.
     @param derivType

     @see FillCache2
     */
    template<typename PointType>
    KOKKOS_FUNCTION void FillCache1(double*          polyCache,
                                    PointType const& pt,
                                    DerivativeFlags::DerivativeType derivType) const
    {
        // Fill in first derivative information if needed
        if((derivType == DerivativeFlags::Input)||(derivType==DerivativeFlags::MixedInput)){
            for(unsigned int d=0; d<dim_-1; ++d)
                basis1d_.EvaluateDerivatives(d, &polyCache[startPos_(d)],&polyCache[startPos_(d+dim_)], maxDegrees_(d), pt(d));

        // Evaluate all degrees of all 1d polynomials except the last dimension, which will be evaluated inside the integrand
        }else{
            for(unsigned int d=0; d<dim_-1; ++d)
                basis1d_.EvaluateAll(d, &polyCache[startPos_(d)], maxDegrees_(d), pt(d));
        }
    }

    /**
     @brief Precomputes parts of the cache that depend on the \f$d^{th}\f$ component of the point \f$x\f$.
     @details
     @tparam PointType The vector type used to define the point.  Can be anything allowing access to components with operator().  Examples are Kokkos::View<double*> or Eigen::VectorXd.  Only the first d-1 components of the vector will be accessed in this function.
     @param polyCache A pointer to the start of the cache.  This memory must be allocated before calling this function.
     @param pt The point to use when filling in the cache.  Should contain \f$[x_1,\ldots,x_{d-1}]\f$.  The vector itself can have more than \f$d-1\f$ components, but only the first \f$d-1\f$ will be accessed.  The value of \f$x_d\f$ is passed through the `xd` argument.
     @param xd The value of \f$x_d\f$.  This is passed separate from \f$[x_1,\ldots,x_{d-1}]\f$ to make integrating over the last component more efficient.  A copy of the point does not need to be created.
     @param derivType

     @see FillCache1
     */

    template<typename PointType>
    KOKKOS_FUNCTION void FillCache2(double*          polyCache,
                                    PointType const&,
                                    double           xd,
                                    DerivativeFlags::DerivativeType derivType) const
    {

        if((derivType==DerivativeFlags::None)||(derivType==DerivativeFlags::Parameters)){

            basis1d_.EvaluateAll(dim_-1,
                                 &polyCache[startPos_(dim_-1)],
                                 maxDegrees_(dim_-1),
                                 xd);

        }else if((derivType==DerivativeFlags::Diagonal) || (derivType==DerivativeFlags::Input)){
            basis1d_.EvaluateDerivatives(dim_ - 1,                          // input dimension
                                         &polyCache[startPos_(dim_-1)],     // basis vals
                                         &polyCache[startPos_(2*dim_-1)],   // basis derivatives
                                         maxDegrees_(dim_-1),               // largest basis degree
                                         xd);                               // point to evaluate at

        }else if((derivType==DerivativeFlags::Diagonal2) || (derivType==DerivativeFlags::MixedInput)){
            basis1d_.EvaluateSecondDerivatives(dim_ - 1,                          // input dimension
                                               &polyCache[startPos_(dim_-1)],     // basis vals
                                               &polyCache[startPos_(2*dim_-1)],   // basis derivatives
                                               &polyCache[startPos_(2*dim_)],     // basis second derivatives
                                               maxDegrees_(dim_-1),               // largest basis degree
                                               xd);                               // point to evaluate at
        }
    }


    template<typename CoeffVecType>
    KOKKOS_FUNCTION double Evaluate(const double* polyCache, CoeffVecType const& coeffs) const
    {
        const unsigned int numTerms = multiSet_.Size();

        double output = 0.0;
        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {
            // Check if the term is constant
            if(multiSet_.nzStarts(termInd)==multiSet_.nzStarts(termInd+1)) {
                output += coeffs(termInd);
                continue;
            }
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            // Need to guard against when the term is constant
            unsigned int end_idx = multiSet_.nzStarts(termInd+1)-1;
            for(unsigned int i=multiSet_.nzStarts(termInd); i<end_idx; ++i) {
                termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
            }

            if constexpr(!std::is_same_v<Rectifier,Identity>){
                if(multiSet_.nzDims(end_idx)==dim_-1){
                    termVal = Rectifier::Evaluate(termVal);
                }
            }
            termVal *= polyCache[startPos_(multiSet_.nzDims(end_idx)) + multiSet_.nzOrders(end_idx)];

            output += termVal*coeffs(termInd);
        }

        return output;
    }

    /**
     * @brief Evaluates the derivative of the expansion wrt x_{D-1}
     *
     * @tparam CoeffVecType
     * @param polyCache
     * @param coeffs
     * @return double
     */
    template<typename CoeffVecType>
    KOKKOS_FUNCTION double DiagonalDerivative(const double* polyCache, CoeffVecType const& coeffs, unsigned int derivOrder) const
    {
        if((derivOrder==0)||(derivOrder>2)){
            assert((derivOrder==1)||(derivOrder==2));
        }

        const unsigned int numTerms = multiSet_.Size();
        double output = 0.0;

        const unsigned int posIndex = 2*dim_+derivOrder-2;

        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            bool hasDeriv = false;
            for(unsigned int i=multiSet_.nzStarts(termInd); i<multiSet_.nzStarts(termInd+1); ++i){
                if(multiSet_.nzDims(i)==dim_-1){
                    if constexpr(!std::is_same_v<Rectifier,Identity>){
                        termVal = Rectifier::Evaluate(termVal);
                    }
                    termVal *= polyCache[startPos_(posIndex) + multiSet_.nzOrders(i)];
                    hasDeriv = true;
                }else{
                    termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
                }
            }
            if(hasDeriv){
                // Multiply by the coefficients to get the contribution to the output
                output += termVal*coeffs(termInd);
            }
        }

        return output;
    }

    /**
     * @brief Evaluates the expansion and also computes the gradient of the expansion output wrt the coefficients.
       @details Using cached values in the "polyCache" argument and coefficients \f$\theta\f$ from the coeffs argument,
        this function returns the value of the expansion \f$f(x;\theta)\f$ and computes the gradient \f$\nabla_\theta f\f$
        of the expansion output with respect to the coefficients \f$\theta\f$.

     @tparam CoeffVecType
     @tparam GradVecType
     @param polyCache
     @param coeffs
     @param grad A vector that will be updated with the scaled gradient.  This is the vector \f$g\f$ in the expression above.
     @param gradScale The scaling \f$\alpha\f$ used in the expression above.
     */
    template<typename CoeffVecType, typename GradVecType>
    KOKKOS_FUNCTION double CoeffDerivative(const double* polyCache, CoeffVecType const& coeffs, GradVecType& grad) const
    {
        const unsigned int numTerms = multiSet_.Size();
        double f=0;

        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {
            // Check if the term is constant
            if(multiSet_.nzStarts(termInd)==multiSet_.nzStarts(termInd+1)) {
                f += coeffs(termInd);
                grad(termInd) = 1.0;
                continue;
            }
            // Compute the value of this term in the expansion
            double termVal = 1.0;
            unsigned int end_idx = multiSet_.nzStarts(termInd+1)-1;
            for(unsigned int i=multiSet_.nzStarts(termInd); i<end_idx; ++i)
                termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
            
            if constexpr(!std::is_same_v<Rectifier,Identity>){
                if(multiSet_.nzDims(end_idx)==dim_-1){
                    termVal = Rectifier::Evaluate(termVal);
                }
            }
            termVal *= polyCache[startPos_(multiSet_.nzDims(end_idx)) + multiSet_.nzOrders(end_idx)];
            f += termVal*coeffs(termInd);
            grad(termInd) = termVal;
        }

        return f;
    }

    /** Computes the gradient \f$\nabla_x f(x_{1:d})\f$ with respect to the input \f$x\f$.
        @param polyCache[in] Cache vector that has been set up by calling both FillCache1 and FillCache2 with `DerivativeFlags::Input`
        @param coeffs[in] Vector of coefficients.  Must have parentheses access operator.
        @param grad[out] Preallocated vector to hold the gradient.
        @return The value of $f(x_{1:d})$.
    */
    template<typename CoeffVecType, typename GradVecType>
    KOKKOS_FUNCTION double InputDerivative(const double* polyCache, CoeffVecType const& coeffs, GradVecType& grad) const
    {
        const unsigned int numTerms = multiSet_.Size();
        double f = 0.0;

        for(int wrt=-1; wrt<int(dim_); ++wrt){

            if(wrt>=0){
                grad(wrt) = 0;
            }

            for(unsigned int termInd=0; termInd<numTerms; ++termInd)
            {
                // Check if the term is constant, so deriv is zero
                if(multiSet_.nzStarts(termInd)==multiSet_.nzStarts(termInd+1)) {
                    if(wrt<0) f += coeffs(termInd);
                    continue;
                }
                double termVal = GetTermValInputDeriv(termInd, wrt, polyCache);
                if(wrt<0){
                    f += termVal*coeffs(termInd);
                } else {
                    // Multiply by the coefficients to get the contribution to the output
                    grad(wrt) += termVal*coeffs(termInd);
                }
            }
        }
        return f;
    }

    /** Computes the gradient of the diagonal derivative \f$\nabla_x \partial_d f(x_{1:d})\f$ with respect to the input \f$x\f$.
        @param polyCache[in] Cache vector that has been set up by calling both FillCache1 and FillCache2 with `DerivativeFlags::MixedInput`
        @param coeffs[in] Vector of coefficients.  Must have parentheses access operator.
        @param grad[out] Preallocated vector to hold the gradient.
        @return The value of $\partial_d f(x_{1:d})$.
    */
    template<typename CoeffVecType, typename GradVecType>
    KOKKOS_FUNCTION double MixedInputDerivative(const double* polyCache, CoeffVecType const& coeffs, GradVecType& grad) const
    {
        const unsigned int numTerms = multiSet_.Size();
        double df = 0.0;
        unsigned int posInd;

        for(int wrt=-1; wrt<int(dim_); ++wrt){
            if(wrt>=0){
                grad(wrt) = 0;
            }

            for(unsigned int termInd=0; termInd<numTerms; ++termInd)
            {
                // Check if the term is constant, so deriv is zero
                if(multiSet_.nzStarts(termInd)==multiSet_.nzStarts(termInd+1)) {
                    continue;
                }
                double termVal = GetTermValMixedDeriv(termInd, wrt, polyCache);

                // Multiply by the coefficients to get the contribution to the output
                // Suppose termVal = 0. if wrt or dim is not used in termInd
                if(wrt<0){
                    df += termVal*coeffs(termInd);
                } else {
                    grad(wrt) += termVal*coeffs(termInd);
                }
            }
        }
        return df;
    }

    /** Computes the gradient of the diagonal derivative \f$\nabla_w \partial_d f(x_1:d; w)\f$ with respect to the parameters.
        @param polyCache[in] Cache vector that has been set up by calling both FillCache1 and FillCache2 with `DerivativeFlags::Mixed`
        @param coeffs[in] Vector of coefficients.  Must have parentheses access operator.
        @param grad[out] Preallocated vector to hold the gradient.
        @return The value of $\partial_d f(x_{1:d})$.
    */
    template<typename CoeffVecType, typename GradVecType>
    KOKKOS_FUNCTION double MixedCoeffDerivative(const double* cache, CoeffVecType const& coeffs, unsigned int derivOrder, GradVecType& grad) const
    {
        const unsigned int numTerms = multiSet_.Size();

        if((derivOrder==0)||(derivOrder>2)){
            assert((derivOrder==1) || (derivOrder==2));
        }

        double df=0;

        const unsigned int posIndex = 2*dim_+derivOrder-2;

        // Compute coeff * polyval for each term
        for(unsigned int termInd=0; termInd<numTerms; ++termInd)
        {
            double termVal = GetTermValMixedCoeffDeriv(termInd, derivOrder, cache, posIndex);
            // Multiply by the coefficients to get the contribution to the output
            df += termVal*coeffs(termInd);
            grad(termInd) = termVal;
        }

        return df;
    }

    /** Allows access to the Fixed MultiIndex Set
     * @return The Fixed MultiIndex Set
     */
    FixedMultiIndexSet<MemorySpace> GetMultiIndexSet() const { return multiSet_; }

#if defined(MPART_HAS_CEREAL)
    template<typename Archive>
    void save(Archive& ar) const{
        ar(dim_, multiSet_, basis1d_);
        ar(startPos_, cacheSize_);
    }

    template<typename Archive>
    void load(Archive& ar) {
        ar(dim_, multiSet_, basis1d_);
        ar(startPos_, cacheSize_);
        maxDegrees_ = multiSet_.MaxDegrees();
    }
#endif // MPART_HAS_CEREAL

private:

    KOKKOS_FUNCTION double GetTermValInputDeriv(unsigned int termInd, int wrt, const double* polyCache) const {
        double termVal = 1.0; // Compute the value of this term in the expansion
        double wrtVal = 0.; // value of univariate basis fcn in dim wrt
        double wrtDeriv = 0.; // derivative of univariate basis fcn in dim wrt
        unsigned int end_idx = multiSet_.nzStarts(termInd+1)-1; // last index in loop
        for(unsigned int i=multiSet_.nzStarts(termInd); i < end_idx; ++i){
            if(int(multiSet_.nzDims(i))==wrt){
                // wrt is the dimension we diff with respect to
                wrtVal = polyCache[startPos_(wrt) + multiSet_.nzOrders(i)];
                // We add dim_ to get the derivative index
                wrtDeriv = polyCache[startPos_(dim_+wrt) + multiSet_.nzOrders(i)];
            }else{
                termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
            }
        }
        unsigned int d = multiSet_.nzDims(end_idx); // Last index
        if(d == wrt) {
            wrtDeriv = polyCache[startPos_(dim_+wrt) + multiSet_.nzOrders(end_idx)];
            if constexpr(!std::is_same_v<Rectifier,Identity>){
                if(d == dim_-1){ // Derivative wrt last input
                    termVal = Rectifier::Evaluate(termVal)*wrtDeriv;
                } else { // no midx entry on last input
                    wrtVal = polyCache[startPos_(wrt) + multiSet_.nzOrders(end_idx)];
                    termVal = Rectifier::Derivative(termVal*wrtVal)*termVal*wrtDeriv;
                }
            } else { // Reduce to loop body
                termVal *= wrtDeriv;
            }
        } else { // If we've "already" differentiated
            double lastVal = polyCache[startPos_(d) + multiSet_.nzOrders(end_idx)];
            if constexpr(!std::is_same_v<Rectifier,Identity>){
                // What value to rectify? if d == dim, we add lastVal
                double rectifyVal = d == dim_-1 ? termVal*wrtVal : termVal*lastVal*wrtVal;
                if(wrt == -1) { // If we're not differentiating
                    double diagVal = d == dim_-1 ? lastVal : 1.;
                    termVal = Rectifier::Evaluate(rectifyVal)*diagVal;
                } else { // If we are differentiating, d_x f(p(x))g(y) = f'(p(x))p'(x)g(y)
                    // Similarly d_x f(p(x)q(y)) = f'(p(x)q(y))p'(x)q(y)
                    // Note wrtDeriv is init'd as 0, so this is zero if midx doesn't use dimension wrt
                    termVal = Rectifier::Derivative(rectifyVal)*termVal*wrtDeriv*lastVal;
                }
            } else { // Reduce to loop body, more-or-less
                termVal *=  wrt == -1 ? lastVal : lastVal*wrtDeriv;
            }
        }
        return termVal;
    }

    KOKKOS_FUNCTION double GetTermValMixedDeriv(unsigned int termInd, int wrt, const double* polyCache) const {
        // Compute the value of this term in the expansion
        double termVal = 1.0;
        double wrtVal = 0.; // value of univariate basis fcn in dim wrt
        double wrtDeriv = (wrt == int(dim_)-1 || wrt == -1); // derivative of univariate basis fcn in dim wrt
        unsigned int end_idx = multiSet_.nzStarts(termInd+1)-1; // last index in loop
        
        if(multiSet_.nzDims(end_idx)!=dim_-1) return 0.; // Value is zero if constant in last dimension

        for(int i=multiSet_.nzStarts(termInd); i<end_idx; ++i){
            if(int(multiSet_.nzDims(i))==wrt){
                wrtVal = polyCache[startPos_(wrt) + multiSet_.nzOrders(i)];
                wrtDeriv = polyCache[startPos_(dim_+wrt) + multiSet_.nzOrders(i)];
            }else{
                termVal *= polyCache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
            }
        }
        int posInd = (wrt==int(dim_)-1) ? (2*dim_) : (2*dim_-1);
        double diagVal = polyCache[startPos_(posInd) + multiSet_.nzOrders(end_idx)];

        if constexpr(!std::is_same_v<Rectifier,Identity>){
            termVal = (wrt==int(dim_)-1 || wrt == -1) ? Rectifier::Evaluate(termVal) : Rectifier::Derivative(termVal*wrtVal);
        }
        termVal *= (diagVal*wrtDeriv);
        return termVal;
    }

    KOKKOS_FUNCTION double GetTermValMixedCoeffDeriv(unsigned int termInd, int wrt, const double* cache, unsigned int posIndex) const {
        // Compute the value of this term in the expansion
        double termVal = 1.0;
        bool hasDeriv = false;
        if(multiSet_.nzStarts(termInd)==multiSet_.nzStarts(termInd+1)) return 0.; // Value is zero if constant in last dimension
        unsigned int end_idx = multiSet_.nzStarts(termInd+1)-1; // last index in loop
        if(multiSet_.nzDims(end_idx)!=dim_-1) return 0.; // Value is zero if constant in last dimension
        for(unsigned int i=multiSet_.nzStarts(termInd); i<end_idx; ++i){
            termVal *= cache[startPos_(multiSet_.nzDims(i)) + multiSet_.nzOrders(i)];
        }
        double diagVal = cache[startPos_(posIndex) + multiSet_.nzOrders(end_idx)];
        
        if constexpr(!std::is_same_v<Rectifier,Identity>){
            termVal = Rectifier::Evaluate(termVal);   
        }

        termVal *= diagVal;
        return termVal;
    }

    unsigned int dim_;

    FixedMultiIndexSet<MemorySpace> multiSet_;
    BasisEvaluatorType basis1d_;

    Kokkos::View<unsigned int*,MemorySpace> startPos_;
    Kokkos::View<const unsigned int*,MemorySpace> maxDegrees_;

    unsigned int cacheSize_;

}; // class MultivariateExpansion



} // namespace mpart



#endif  // #ifndef MPART_MULTIVARIATEEXPANSION_H