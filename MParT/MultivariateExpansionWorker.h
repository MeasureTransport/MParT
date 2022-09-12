#ifndef MPART_MultivariateExpansionWorker_H
#define MPART_MultivariateExpansionWorker_H

#include <Kokkos_Core.hpp>

#include "MParT/DerivativeFlags.h"

#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/Utilities/ArrayConversions.h"

namespace mpart{

template<typename MemorySpace>
struct MultivariateExpansionMaxDegreeFunctor {

    MultivariateExpansionMaxDegreeFunctor(unsigned int dim, Kokkos::View<unsigned int*, MemorySpace> startPos, Kokkos::View<const unsigned int*, MemorySpace> maxDegreesIn);


    KOKKOS_FUNCTION void operator()(const unsigned int i, unsigned int& update, const bool final) const;

    unsigned int dim;
    Kokkos::View<unsigned int*, MemorySpace> startPos;
    Kokkos::View<const unsigned int*, MemorySpace> maxDegrees;
};

template<typename MemorySpace>
struct CacheSizeFunctor{

    CacheSizeFunctor(Kokkos::View<unsigned int*, MemorySpace> startPosIn, Kokkos::View<unsigned int*, MemorySpace> cacheSizeIn);

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
     \vdots
     \phi_1^{p_1}\\
     \phi_2^0(x_2)\\
     \vdots\\
     \phi_2^{p_2}(x_2)\\
     \vdots \\
     \phi_d^{p_d}(x_d)\\
     \frac{\partial}{\partial x_d}\phi_d^0(x_d)\\
     \vdots\\
     \frac{\partial}{\partial x_d}\phi_d^{p_d}(x_d)\\
     \frac{\partial^2}{\partial x_d}\phi_d^0(x_d^2)\\
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

    MultivariateExpansionWorker();

    MultivariateExpansionWorker(MultiIndexSet const& multiSet,
                                BasisEvaluatorType const& basis1d = BasisEvaluatorType());

    MultivariateExpansionWorker(FixedMultiIndexSet<MemorySpace> const& multiSet,
                                BasisEvaluatorType const& basis1d = BasisEvaluatorType());

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
    KOKKOS_FUNCTION void FillCache1(double*          polyCache,
                                    StridedVector<const double, MemorySpace> pt,
                                    DerivativeFlags::DerivativeType derivType) const;

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
    KOKKOS_FUNCTION void FillCache2(double*          polyCache,
                                    StridedVector<const double, MemorySpace>,
                                    double           xd,
                                    DerivativeFlags::DerivativeType derivType) const;


    KOKKOS_FUNCTION double Evaluate(const double* polyCache, StridedVector<const double, MemorySpace> coeffs) const;

    /**
     * @brief Evaluates the derivative of the expansion wrt x_{D-1}
     *
     * @tparam CoeffVecType
     * @param polyCache
     * @param coeffs
     * @return double
     */
    KOKKOS_FUNCTION double DiagonalDerivative(const double* polyCache, StridedVector<const double, MemorySpace>, unsigned int derivOrder) const;

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
    KOKKOS_FUNCTION double CoeffDerivative(const double* polyCache, 
                                           StridedVector<const double, MemorySpace> coeffs, 
                                           StridedVector<double, MemorySpace> grad) const;

    /** Computes the gradient \f$\nabla_x f(x_{1:d})\f$ with respect to the input \f$x\f$. 
        @param polyCache[in] Cache vector that has been set up by calling both FillCache1 and FillCache2 with `DerivativeFlags::Input`
        @param coeffs[in] Vector of coefficients.  Must have parentheses access operator.
        @param grad[out] Preallocated vector to hold the gradient. 
        @return The value of $f(x_{1:d})$.
    */
    KOKKOS_FUNCTION double InputDerivative(const double* polyCache, 
                                           StridedVector<const double, MemorySpace> coeffs, 
                                           StridedVector<double, MemorySpace> grad) const;

    /** Computes the gradient of the diagonal derivative \f$\nabla_x \partial_d f(x_{1:d})\f$ with respect to the input \f$x\f$. 
        @param polyCache[in] Cache vector that has been set up by calling both FillCache1 and FillCache2 with `DerivativeFlags::MixedInput`
        @param coeffs[in] Vector of coefficients.  Must have parentheses access operator.
        @param grad[out] Preallocated vector to hold the gradient. 
        @return The value of $\partial_d f(x_{1:d})$.
    */
    KOKKOS_FUNCTION double MixedInputDerivative(const double* polyCache, 
                                                StridedVector<const double, MemorySpace> coeffs,
                                                StridedVector<double, MemorySpace> grad) const;

    /** Computes the gradient of the diagonal derivative \f$\nabla_w \partial_d f(x_1:d; w)\f$ with respect to the parameters. 
        @param polyCache[in] Cache vector that has been set up by calling both FillCache1 and FillCache2 with `DerivativeFlags::Mixed`
        @param coeffs[in] Vector of coefficients.  Must have parentheses access operator.
        @param grad[out] Preallocated vector to hold the gradient. 
        @return The value of $\partial_d f(x_{1:d})$.
    */
    KOKKOS_FUNCTION double MixedCoeffDerivative(const double* cache, 
                                                StridedVector<const double, MemorySpace> coeffs, 
                                                unsigned int derivOrder, 
                                                StridedVector<double, MemorySpace> grad) const;

private:

    unsigned int dim_;

    FixedMultiIndexSet<MemorySpace> multiSet_;
    BasisEvaluatorType basis1d_;

    Kokkos::View<unsigned int*,MemorySpace> startPos_;
    Kokkos::View<const unsigned int*,MemorySpace> maxDegrees_;

    unsigned int cacheSize_;

}; // class MultivariateExpansion



} // namespace mpart



#endif  // #ifndef MPART_MULTIVARIATEEXPANSION_H