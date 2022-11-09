#include "MParT/Distributions/DensityBase.h"

using namespace mpart;

template<>
Eigen::VectorXd DensityBase<Kokkos::HostSpace>::LogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts) {
    // Allocate output
    Eigen::VectorXd output(pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double, Kokkos::HostSpace>(pts);
    StridedVector<double, Kokkos::HostSpace> outView = VecToKokkos<double, Kokkos::HostSpace>(output);
    // Call the LogDensity function
    LogDensityImpl(ptsView, outView);

    return output;
}

template<>
Eigen::RowMatrixXd DensityBase<Kokkos::HostSpace>::GradLogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts) {
    // Allocate output
    Eigen::RowMatrixXd output(dimension, pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double, Kokkos::HostSpace>(pts);
    StridedMatrix<double, Kokkos::HostSpace> outView = MatToKokkos<double, Kokkos::HostSpace>(output);
    // Call the LogDensity function
    GradLogDensityImpl(ptsView, outView);

    return output;
}