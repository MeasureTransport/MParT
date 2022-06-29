#ifndef MPART_IDENTITYMAP_H
#define MPART_IDENTITYMAP_H

#include "MParT/ConditionalMapBase.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>


namespace mpart{

/**
 @brief Provides a definition of the identity map.
 @details 
This class defines the identity map \f$I:\mathbb{R}^N\rightarrow \mathbb{R}^N\f$ such that \f$I(x) = x\f$ 


 */
class IdentityMap : public ConditionalMapBase
{
public:

    /** @brief Construct a block triangular map from a collection of other ConditionalMapBase objects.

    @param dim The dimension of the map \f$N\f$
    */
    IdentityMap(unsigned int dim);

    // /** @brief Computes the log determinant of the Jacobian matrix of this map.

    // @details
    // @param pts The points where the jacobian should be evaluated.  Should have \f$N\f$ rows.  Each column contains a single point.
    // @param output A vector with length equal to the number of columns in pts containing the log determinant of the Jacobian.  This 
    //               vector should be correctly allocated and sized before calling this function.
    // */
    // virtual void LogDeterminantImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
    //                                 Kokkos::View<double*, Kokkos::HostSpace>             &output) override;


    // /** @brief Evaluates the map.

    // @details
    // @param pts The points where the map should be evaluated.  Should have \f$N\f$ rows.  Each column contains a single point.
    // @param output A matrix with \f$M\f$ rows to store the map output.  Should have the same number of columns as pts and be 
    //               allocated before calling this function.
    // */
    // void EvaluateImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
    //                   Kokkos::View<double**, Kokkos::HostSpace>            & output) override;


    // /** @brief Evaluates the map inverse.

    // @details To understand this function, consider splitting the map input \f$x_{1:N}\f$ into two parts so that \f$x_{1:N} = [x_{1:N-M},x_{N-M+1:M}]\f$.  Note that the 
    // second part of this split \f$x_{N-M+1:M}\f$ has the same dimension as the map output.   With this split, the map becomes 
    // \f$T(x_{1:N-M},x_{N-M+1:M})=r_{1:M}\f$.  Given \f$x_{1:N-M}\f$ and \f$r_{1:M}\f$, the `InverseImpl` function solves for the value 
    // of \f$x_{N-M+1:M}\f$ that satisfies \f$T(x_{1:N-M},x_{N-M+1:M})=r_{1:M}\f$.   Our shorthand for this will be 
    // \f$x_{N-M+1:M} = T^{-1}(r_{1:M}; x_{1:N-M})\f$.
    
    // @param x1 The points \f$x_{1:N-M}\f$ where the map inverse should be evaluated.  Each column contains a single point.
    // @param r The map output \f$r_{1:M}\f$ to invert.  
    // @param output A matrix with \f$M\f$ rows to store the computed map inverse \f$x_{N-M+1:M}\f$.
    // */
    // virtual void InverseImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& x1, 
    //                          Kokkos::View<const double**, Kokkos::HostSpace> const& r,
    //                          Kokkos::View<double**, Kokkos::HostSpace>            & output) override;


    // virtual void InverseInplace(Kokkos::View<double**, Kokkos::HostSpace> const& x1, 
    //                             Kokkos::View<const double**, Kokkos::HostSpace> const& r);

// private:

//     std::vector<std::shared_ptr<ConditionalMapBase>> comps_;


}; // class IdentityMap

}

#endif