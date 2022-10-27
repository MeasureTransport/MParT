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
void GaussianDistribution<MemorySpace>::LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) {
    // Compute the log density
    StridedMatrix<double, MemorySpace> diff;
    int M = pts.extent(0);
    int N = pts.extent(1);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{N, M}});
    if(mean_.extent(0) == 0){
        diff = pts;
    }
    else {
        diff = StridedMatrix<double, MemorySpace>("diff", M, N);
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& j, const int& i) {
            diff(i,j) = pts(i,j) - mean_(i);
        });
    }

    if(!idCov_) {
        covLU_.solveInPlace(diff);
    }

    Kokkos::parallel_for( "log terms", N, KOKKOS_LAMBDA (const int& j) {
        output(j) = -0.5*( M*logtau_ + logDet_ );
    });

    Kokkos::parallel_for( "weighted norm", policy, KOKKOS_LAMBDA (const int& j, const int& i) {
        output(j) -= 0.5*diff(i,j)*diff(i,j);
    });
}

void GaussianDistribution<MemorySpace>::GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) {

}