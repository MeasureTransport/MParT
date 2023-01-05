#include "MParT/ConditionalMapBase.h"
#include "MParT/InverseMap.h"

// Just use inverse of forward map
template<typename MemorySpace>
void InverseMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                  StridedMatrix<double, MemorySpace> output) {
    // Assume square
    StridedMatrix<const double, MemorySpace> x1;
    fmap_->InverseImpl(x1, pts, output);
}

// Just use evaluate of forward map
template<typename MemorySpace>
void InverseMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const&,
                    StridedMatrix<const double, MemorySpace> const& r,
                    StridedMatrix<double, MemorySpace> output) {
    fmap_->EvaluateImpl(r, output);
}

// Just use negative of forward logdet
template<typename MemorySpace>
void InverseMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                        StridedVector<double, MemorySpace> output) {
    fmap_->LogDeterminantImpl(pts, output);
    Kokkos::parallel_for("InverseMap::LogDeterminantImpl", output.extent(0), KOKKOS_LAMBDA(int i) {
        output(i) = -output(i);
    });
}

// Perform Matrix solve with jacobian of forward map wrt coeffs
template<typename MemorySpace>
void InverseMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                    StridedMatrix<const double, MemorySpace> const& sens,
                    StridedMatrix<double, MemorySpace> output) {
    // TODO: Implement
}

// Perform Matrix solve with jacobian of forward map wrt inputs
template<typename MemorySpace>
void InverseMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                    StridedMatrix<const double, MemorySpace> const& sens,
                    StridedMatrix<double, MemorySpace> output) {
    // TODO: Implement
}

// Negative of forward logdet coeff grad
template<typename MemorySpace>
void InverseMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                    StridedMatrix<double, MemorySpace> output) {
    fmap_->LogDeterminantCoeffGradImpl(pts, output);
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {output.extent(0), output.extent(1)});
    Kokkos::parallel_for("InverseMap::LogDeterminantCoeffGradImpl", policy, KOKKOS_LAMBDA(int i, int j) {
        output(i, j) = -output(i, j);
    });
}

// Negative of forward logdet input grad
template<typename MemorySpace>
void InverseMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                    StridedMatrix<double, MemorySpace> output) {
    fmap_->LogDeterminantInputGradImpl(pts, output);
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {output.extent(0), output.extent(1)});
    Kokkos::parallel_for("InverseMap::LogDeterminantInputGradImpl", policy, KOKKOS_LAMBDA(int i, int j) {
        output(i, j) = -output(i, j);
    });
}