#include "MParT/Distributions/DensityBase.h"

using namespace mpart;

template<>
Eigen::VectorXd DensityBase<Kokkos::HostSpace>::LogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts) {
    // Allocate output
    Eigen::VectorXd out(pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double, Kokkos::HostSpace>(pts);
    StridedVector<double, Kokkos::HostSpace> outView = VecToKokkos<double, Kokkos::HostSpace>(output);
    // Call the LogDensity function
    auto output_kokkos = LogDensityImpl(ptsView, outView);

    return out;
}

template<>
Eigen::RowMatrixXd DensityBase<Kokkos::HostSpace>::GradLogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts) {
    // Allocate output
    Eigen::RowMatrixXd out(dimension, pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double, Kokkos::HostSpace>(pts);
    StridedMatrix<double, Kokkos::HostSpace> outView = RowMatToKokkos<double, Kokkos::HostSpace>(output);
    // Call the LogDensity function
    auto output_kokkos = GradLogDensityImpl(ptsView, outView);

    return out;
}