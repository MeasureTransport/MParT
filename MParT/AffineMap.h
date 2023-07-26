#ifndef MPART_AFFINEMAP_H
#define MPART_AFFINEMAP_H

#include <Kokkos_Core.hpp>

#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/GPUtils.h"
#include "MParT/Utilities/LinearAlgebra.h"

#include <Eigen/Core>
#include <Eigen/Dense>

namespace mpart{

/** @brief Defines transformations of the form \f$Ax+b\f$ for an invertible matrix \f$A\f$ and vector offset \f$b\f$.
 * Makes a deep copy of any views passed to the constructor.
*/
template<typename MemorySpace>
class AffineMap : public ConditionalMapBase<MemorySpace>
{
public:
    /** Construct an affine map that simply shifts the input.
    \f[ y = x + b \f]
    */
    AffineMap(StridedVector<double,MemorySpace> b);

    /** Constructs a linear map that only scales the input.  Note that the matrix A must be invertible.
    \f[ y = Ax\f]
    */
    AffineMap(StridedMatrix<double,MemorySpace> A);

    /** Constructs an affine map that scales and shifts the input. Note that the matrix A must be invertible.
    \f[ y = ax + b \f]
    */
    AffineMap(StridedMatrix<double,MemorySpace> A, StridedVector<double,MemorySpace> b);

    virtual ~AffineMap() = default;

    void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                            StridedVector<double, MemorySpace>              output) override;

    void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                     StridedMatrix<const double, MemorySpace> const& r,
                     StridedMatrix<double, MemorySpace>              output) override;

    void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                     StridedMatrix<double, MemorySpace>              output) override;

    void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<double, MemorySpace>              output) override;

    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                       StridedMatrix<const double, MemorySpace> const& sens,
                       StridedMatrix<double, MemorySpace>              output) override;

    void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<const double, MemorySpace> const& sens,
                      StridedMatrix<double, MemorySpace>              output) override;

    void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                     StridedMatrix<double, MemorySpace>              output) override;

    /** Computes an LU factorization of the matrix A_ */
    void Factorize();

protected:

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> A_;
    Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace> b_;

    mpart::PartialPivLU<MemorySpace> luSolver_;
    double logDet_;
};

}
#endif