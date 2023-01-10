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
Eigen::RowMatrixXd DensityBase<Kokkos::HostSpace>::LogDensityInputGrad(Eigen::Ref<const Eigen::RowMatrixXd> const &pts) {
    // Allocate output
    Eigen::RowMatrixXd output(pts.rows(), pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double, Kokkos::HostSpace>(pts);
    StridedMatrix<double, Kokkos::HostSpace> outView = MatToKokkos<double, Kokkos::HostSpace>(output);
    // Call the LogDensity function
    LogDensityInputGradImpl(ptsView, outView);

    return output;
}

template<>
template<>
StridedVector<double, Kokkos::HostSpace> DensityBase<Kokkos::HostSpace>::LogDensity<Kokkos::HostSpace>(StridedMatrix<const double, Kokkos::HostSpace> const &X) {
    // Allocate output
    Kokkos::View<double*, Kokkos::HostSpace> output("output", X.extent(1));
    // Call the LogDensity function
    LogDensityImpl(X, output);

    return output;
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> DensityBase<Kokkos::HostSpace>::LogDensityInputGrad<Kokkos::HostSpace>(StridedMatrix<const double, Kokkos::HostSpace> const &X) {
    // Allocate output
    Kokkos::View<double**, Kokkos::HostSpace> output("output", X.extent(0), X.extent(1));
    // Call the LogDensity function
    LogDensityInputGradImpl(X, output);

    return output;
}