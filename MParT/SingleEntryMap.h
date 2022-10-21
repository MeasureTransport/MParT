#ifndef MPART_SINGLEENTRYMAP_H
#define MPART_SINGLEENTRYMAP_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/Miscellaneous.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>


namespace mpart{

/**
 @brief Provides a definition of a single entry transport map.
 @details
This class defines a map \f$T:\mathbb{R}^N\rightarrow \mathbb{R}^M\f$ that is the identity map in all but one component
\f[
T(x) = \left[\begin{array}{l} x_{< i} \\ T_i(x_{\le i}) \\ x_{> i} \end{array} \right],
\f]
where \f$i\f$ is the active index.

 */
template<typename MemorySpace>
class SingleEntryMap : public ConditionalMapBase<MemorySpace>{
        
public:

    /** @brief Construct a single entry transport map.

     
    @param dim  The full dimension of the map
    @param activeInd  The index for the non-identity entry
    @param component  A ConditionalMapBase object defining the non-identity entry of the map.
    */
    SingleEntryMap(const unsigned int dim, const unsigned int activeInd, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& component);

    using ConditionalMapBase<MemorySpace>::SetCoeffs;
    virtual void SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs) override;
    virtual void WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs) override;
    #if defined(MPART_ENABLE_GPU)
    virtual void SetCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs) override;
    virtual void WrapCoeffs(Kokkos::View<double*, mpart::DeviceSpace> coeffs) override;
    #endif 
    
    /** @brief Computes the log determinant of the Jacobian matrix of this map.

    @details
    @param pts The points where the jacobian should be evaluated.  Should have \f$N\f$ rows.  Each column contains a single point.
    @param output A vector with length equal to the number of columns in pts containing the log determinant of the Jacobian.  This
                  vector should be correctly allocated and sized before calling this function.
    */
    virtual void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                    StridedVector<double, MemorySpace>              output) override;


    /** @brief Evaluates the map.

    @details
    @param pts The points where the map should be evaluated.  Should have \f$N\f$ rows.  Each column contains a single point.
    @param output A matrix with \f$M\f$ rows to store the map output.  Should have the same number of columns as pts and be
                  allocated before calling this function.
    */
    void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<double, MemorySpace>              output) override;

    virtual void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                              StridedMatrix<const double, MemorySpace> const& sens,
                              StridedMatrix<double, MemorySpace>              output) override;
    
    /** @brief Evaluates the map inverse.

    @details To understand this function, consider splitting the map input \f$x_{1:N}\f$ into two parts so that \f$x_{1:N} = [x_{1:N-M},x_{N-M+1:M}]\f$.  Note that the
    second part of this split \f$x_{N-M+1:M}\f$ has the same dimension as the map output.   With this split, the map becomes
    \f$T(x_{1:N-M},x_{N-M+1:M})=r_{1:M}\f$.  Given \f$x_{1:N-M}\f$ and \f$r_{1:M}\f$, the `InverseImpl` function solves for the value
    of \f$x_{N-M+1:M}\f$ that satisfies \f$T(x_{1:N-M},x_{N-M+1:M})=r_{1:M}\f$.   Our shorthand for this will be
    \f$x_{N-M+1:M} = T^{-1}(r_{1:M}; x_{1:N-M})\f$.

    @param x1 The points \f$x_{1:N-M}\f$ where the map inverse should be evaluated.  Each column contains a single point.
    @param r The map output \f$r_{1:M}\f$ to invert.
    @param output A matrix with \f$M\f$ rows to store the computed map inverse \f$x_{N-M+1:M}\f$.
    */
    virtual void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                             StridedMatrix<const double, MemorySpace> const& r,
                             StridedMatrix<double, MemorySpace>              output) override;


    virtual void InverseInplace(StridedMatrix<double, MemorySpace>              x1,
                                StridedMatrix<const double, MemorySpace> const& r);


    virtual void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                               StridedMatrix<const double, MemorySpace> const& sens,
                               StridedMatrix<double, MemorySpace>              output) override;


    virtual void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                             StridedMatrix<double, MemorySpace>              output) override;
    
    virtual void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                             StridedMatrix<double, MemorySpace>              output) override;


private:

    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
}; // class SingleEntryMap

}

#endif