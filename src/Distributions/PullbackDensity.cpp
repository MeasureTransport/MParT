#include "MParT/Distributions/PullbackDensity.h"

using namespace mpart;

template<typename MemorySpace>
PullbackDensity<MemorySpace>::PullbackDensity(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<DensityBase<MemorySpace>> density):
    DensityBase<MemorySpace>(density->Dim()), map_(map), density_(density) {
    if (map_->outputDim != density_->Dim()) {
        throw std::invalid_argument("PullbackDensity: map output dimension does not match density dimension");
    }
    if (map_->inputDim != map_->outputDim) {
        throw std::invalid_argument("PullbackDensity: map input dimension does not match map output dimension");
    }
}

template<typename MemorySpace>
void PullbackDensity<MemorySpace>::LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) {
    StridedMatrix<double, MemorySpace> mappedPts = map_->Evaluate(pts);
    density_->LogDensityImpl(mappedPts, output);
    StridedVector<double, MemorySpace> logJacobian = map_->LogDeterminant(pts);
    Kokkos::parallel_for("Add logJacobian", output.extent(0), KOKKOS_LAMBDA(const unsigned int i){output(i) += logJacobian(i);});
}

template<typename MemorySpace>
void PullbackDensity<MemorySpace>::LogDensityInputGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) {
    StridedMatrix<const double, MemorySpace> mappedPts = map_->Evaluate(pts);
    StridedMatrix<double, MemorySpace> sens_map = density_->LogDensityInputGrad(mappedPts);
    map_->GradientImpl(pts, sens_map, output);
    StridedMatrix<double, MemorySpace> gradLogJacobian = map_->LogDeterminantInputGrad(pts);
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>,typename MemoryToExecution<MemorySpace>::Space>({0,0},{output.extent(1),output.extent(0)});
    Kokkos::parallel_for("Add GradLogJacobian",policy,KOKKOS_LAMBDA(const unsigned int j, const unsigned int i){output(i,j) += gradLogJacobian(i,j);});
}

template<typename MemorySpace>
void PullbackDensity<MemorySpace>::LogDensityCoeffGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) {
    StridedMatrix<const double, MemorySpace> mappedPts = map_->Evaluate(pts);
    StridedMatrix<double, MemorySpace> sens_map = density_->LogDensityInputGrad(mappedPts);
    map_->CoeffGradImpl(pts, sens_map, output);
    StridedMatrix<double, MemorySpace> gradLogJacobian = map_->LogDeterminantCoeffGrad(pts);
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>,typename MemoryToExecution<MemorySpace>::Space>({0,0},{output.extent(1),output.extent(0)});
    Kokkos::parallel_for("Add GradLogJacobian",policy,KOKKOS_LAMBDA(const unsigned int j, const unsigned int i){output(i,j) += gradLogJacobian(i,j);});
}

template<typename MemorySpace>
StridedMatrix<double, MemorySpace> PullbackDensity<MemorySpace>::LogDensityCoeffGrad(StridedMatrix<const double, MemorySpace> const &pts) {
    Kokkos::View<double**, MemorySpace> output("LogDensityCoeffGrad", map_->numCoeffs, pts.extent(1));
    LogDensityCoeffGradImpl(pts, output);
    return output;
}

template<>
Eigen::RowMatrixXd PullbackDensity<Kokkos::HostSpace>::LogDensityCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const &pts) {
    // Allocate output
    Eigen::RowMatrixXd output(pts.rows(), pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double, Kokkos::HostSpace>(pts);
    StridedMatrix<double, Kokkos::HostSpace> outView = MatToKokkos<double, Kokkos::HostSpace>(output);
    // Call the LogDensity function
    LogDensityInputGradImpl(ptsView, outView);
    return output;
}

template class mpart::PullbackDensity<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::PullbackDensity<mpart::DeviceSpace>;
#endif