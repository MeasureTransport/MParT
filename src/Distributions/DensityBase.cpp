#include "MParT/Distributions/DensityBase.h"
#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;

template<typename MemorySpace>
Eigen::VectorXd DensityBase<MemorySpace>::LogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts) {
    // Allocate output
    Eigen::VectorXd output(pts.cols());
    StridedVector<double, Kokkos::HostSpace> outView_h = VecToKokkos<double, Kokkos::HostSpace>(output);
    StridedMatrix<const double, MemorySpace> ptsView_d = ConstRowMatToKokkos<double, MemorySpace>(pts);
    StridedVector<double, MemorySpace> outView_d;
    if constexpr(std::is_same_v<MemorySpace, Kokkos::HostSpace>) outView_d = outView_h;
    else outView_d = Kokkos::View<double*, MemorySpace> {"Outview device", ptsView_d.extent(1)};
    // Call the LogDensity function
    LogDensityImpl(ptsView_d, outView_d);
    Kokkos::deep_copy(outView_h, outView_d);
    return output;
}

template<typename MemorySpace>
Eigen::RowMatrixXd DensityBase<MemorySpace>::LogDensityInputGrad(Eigen::Ref<const Eigen::RowMatrixXd> const &pts) {
    // Allocate output
    Eigen::RowMatrixXd output(pts.rows(), pts.cols());
    StridedMatrix<double, Kokkos::HostSpace> outView_h = MatToKokkos<double, Kokkos::HostSpace>(output);
    StridedMatrix<const double, MemorySpace> ptsView_d = ConstRowMatToKokkos<double, MemorySpace>(pts);
    StridedMatrix<double, MemorySpace> outView_d;
    if constexpr(std::is_same_v<MemorySpace, Kokkos::HostSpace>) outView_d = outView_h;
    else outView_d = MatToKokkos<double, MemorySpace>(output); // TODO: Could be optimized
    // Call the LogDensity function
    LogDensityInputGradImpl(ptsView_d, outView_d);
    Kokkos::deep_copy(outView_h, outView_d);
    return output;
}

template<typename MemorySpace>
StridedVector<double, MemorySpace> DensityBase<MemorySpace>::LogDensity(StridedMatrix<const double, MemorySpace> const &X) {
    // Allocate output
    Kokkos::View<double*, MemorySpace> output("output", X.extent(1));
    // Call the LogDensity function
    LogDensityImpl(X, output);
    return output;
}

template<typename MemorySpace>
StridedMatrix<double, MemorySpace> DensityBase<MemorySpace>::LogDensityInputGrad(StridedMatrix<const double, MemorySpace> const &X) {
    // Allocate output
    Kokkos::View<double**, MemorySpace> output("output", X.extent(0), X.extent(1));
    // Call the LogDensity function
    LogDensityInputGradImpl(X, output);

    return output;
}

#if defined(MPART_ENABLE_GPU)

template<>
StridedVector<double, Kokkos::HostSpace> DensityBase<DeviceSpace>::LogDensity(StridedMatrix<const double, Kokkos::HostSpace> const &X) {
    StridedMatrix<const double, DeviceSpace> X_d = ToDevice<DeviceSpace>(X);
    // Allocate output
    Kokkos::View<double*, DeviceSpace> output_d("output", X.extent(1));
    // Call the LogDensity function
    LogDensityImpl(X_d, output_d);
    return ToHost(output_d);
}

template<>
StridedVector<double, DeviceSpace> DensityBase<Kokkos::HostSpace>::LogDensity(StridedMatrix<const double, DeviceSpace> const &X) {
    StridedMatrix<const double, Kokkos::HostSpace> X_h = ToHost(X);
    // Allocate output
    Kokkos::View<double*, Kokkos::HostSpace> output_h("output", X.extent(1));
    // Call the LogDensity function
    LogDensityImpl(X_h, output_h);
    return ToDevice<DeviceSpace>(output_h);
}

template<>
StridedMatrix<double, Kokkos::HostSpace> DensityBase<DeviceSpace>::LogDensityInputGrad(StridedMatrix<const double, Kokkos::HostSpace> const &X) {
    StridedMatrix<const double, DeviceSpace> X_d = ToDevice<DeviceSpace>(X);
    // Allocate output
    Kokkos::View<double**, DeviceSpace> output_d("output", X.extent(0), X.extent(1));
    // Call the LogDensityInputGrad function
    LogDensityInputGradImpl(X_d, output_d);
    return ToHost(output_d);
}

template<>
StridedMatrix<double, DeviceSpace> DensityBase<Kokkos::HostSpace>::LogDensityInputGrad(StridedMatrix<const double, DeviceSpace> const &X) {
    StridedMatrix<const double, Kokkos::HostSpace> X_h = ToHost(X);
    // Allocate output
    Kokkos::View<double**, Kokkos::HostSpace> output_h("output", X.extent(0), X.extent(1));
    // Call the LogDensityInputGrad function
    LogDensityInputGradImpl(X_h, output_h);
    return ToDevice<DeviceSpace>(output_h);
}

#endif

template class mpart::DensityBase<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::DensityBase<mpart::DeviceSpace>;
#endif