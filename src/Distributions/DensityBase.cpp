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