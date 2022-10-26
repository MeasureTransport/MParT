#include "MParT/Distributions/GaussianDistribution.h"

using namespace mpart;

template<typename MemorySpace>
GaussianDistribution<MemorySpace>::GaussianDistribution(StridedVector<double, MemorySpace> mean, StridedMatrix<double, MemorySpace> covar): mean_(mean) {
    Factorize(covar);
}

template<typename MemorySpace>
GaussianDistribution<MemorySpace>::GaussianDistribution(StridedMatrix<double, MemorySpace> covar) {
    Factorize(covar);
}

template<typename MemorySpace>
GaussianDistribution<MemorySpace>::GaussianDistribution(StridedVector<double, MemorySpace> mean): mean_(mean), idCov_(true) {}

template<typename MemorySpace>
StridedVector<double, MemorySpace> GaussianDistribution<MemorySpace>::LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> out) {
    // Compute the log density
    StridedVector<double, MemorySpace> diff;
    if(mean_.extent(0) == 0){
        diff = pts;
    }
    else {
        diff = StridedVector<double, MemorySpace>("diff", pts.extent(0));
        Kokkos::parallel_for("diff", pts.extent(0), KOKKOS_LAMBDA(const int i) {
            diff(i) = pts(i) - mean_(i);
        });
    }

    if(!idCov_) {
        covLU_.solveInPlace(diff);
    }

    double norm = 0.;
    Kokkos::parallel_reduce( "norm", diff.extent(0), KOKKOS_LAMBDA (const int i, double& lsum) {
        lsum += diff(i)*diff(i);
    }, norm);


    auto ptsMinusMeanT = ptsMinusMean.transpose();
    auto covarInv = covLU_.solve(StridedMatrix<double, MemorySpace>(ptsMinusMeanT));
    auto logDensity = -0.5*(ptsMinusMeanT*covarInv).diagonal() - 0.5*mean_.size()*log(tau) - 0.5*logDetCov_;
    // Copy the output
    Kokkos::deep_copy(out, logDensity);
    return out;
}