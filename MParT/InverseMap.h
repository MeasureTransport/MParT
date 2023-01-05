#ifndef MPART_INVERSEMAP_H
#define MPART_INVERSEMAP_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/Miscellaneous.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>

template<typename MemorySpace>
class InverseMap : public ConditionalMapBase<MemorySpace>
{
public:
InverseMap(std::shared_ptr<ConditionalMapBase<MemorySpace> > fmap);

// Just use inverse of forward map
void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                  StridedMatrix<double, MemorySpace>              output) override;

// Just use evaluate of forward map
void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                    StridedMatrix<const double, MemorySpace> const& r,
                    StridedMatrix<double, MemorySpace>              output) override;

// Just use negative of forward logdet
void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                        StridedVector<double, MemorySpace>              output) override;

// Perform Matrix solve with jacobian of forward map wrt coeffs
void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                    StridedMatrix<const double, MemorySpace> const& sens,
                    StridedMatrix<double, MemorySpace>              output) override;

// Perform Matrix solve with jacobian of forward map wrt inputs
void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                    StridedMatrix<const double, MemorySpace> const& sens,
                    StridedMatrix<double, MemorySpace>              output) override;

// I have no idea :/
void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                    StridedMatrix<double, MemorySpace>              output) override;

// I have no idea :/
void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                    StridedMatrix<double, MemorySpace>              output) override;

private:
std::shared_ptr<ConditionalMapBase<MemorySpace> fmap_;
};

#endif // MPART_INVERSEMAP_H