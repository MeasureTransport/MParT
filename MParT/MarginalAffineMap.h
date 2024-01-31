#ifndef MPART_MARGINALMarginalAffineMap_H
#define MPART_MARGINALMarginalAffineMap_H

#include <Kokkos_Core.hpp>

#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/GPUtils.h"

namespace mpart{

/** @brief Map the form \f$T(Dx+b)\f$ for conditional map \f$T\f$, diagonal matrix \f$D\f$ and vector offset \f$b\f$.
 * Makes a deep copy of any views passed to the constructor.
*/
template<typename MemorySpace>
class MarginalAffineMap : public ConditionalMapBase<MemorySpace>
{
public:

    MarginalAffineMap(StridedVector<double,MemorySpace> scale,
    StridedVector<double,MemorySpace> shift,
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map,
    bool moveCoeffs = true);

    virtual ~MarginalAffineMap() = default;

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

protected:

    Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace> scale_;
    Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace> shift_;
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
    double logDet_;
};

}
#endif