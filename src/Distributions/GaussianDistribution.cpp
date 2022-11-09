#include "MParT/Distributions/GaussianDistribution.h"

using namespace mpart;

template<typename MemorySpace>
GaussianDistribution<MemorySpace>::GaussianDistribution(StridedVector<double, MemorySpace> mean, StridedMatrix<double, MemorySpace> covar): mean_(mean), dim_(mean.extent(0)) {
    Factorize(covar);
}

template<typename MemorySpace>
GaussianDistribution<MemorySpace>::GaussianDistribution(StridedMatrix<double, MemorySpace> covar): dim_(covar.extent(0)) {
    Factorize(covar);
}

template<typename MemorySpace>
GaussianDistribution<MemorySpace>::GaussianDistribution(StridedVector<double, MemorySpace> mean): mean_(mean), dim_(mean.extent(0)), idCov_(true) {}

template<typename MemorySpace>
GaussianDistribution<MemorySpace>::GaussianDistribution(): idCov_(true) {}

template<typename MemorySpace>
void GaussianDistribution<MemorySpace>::LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) {
    // Compute the log density
    StridedMatrix<double, MemorySpace> diff;
    int M = pts.extent(0);
    int N = pts.extent(1);
    if(dim_ != 0 && M != dim_) {
        throw std::runtime_error("GaussianDistribution::LogDensityImpl: The number of rows in pts must match the dimension of the distribution.");
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{N, M}});
    diff = StridedMatrix<double, MemorySpace>("diff", M, N);

    if(mean_.extent(0) == 0){
        Kokkos::deep_copy(diff, pts);
    }
    else {
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& j, const int& i) {
            diff(i,j) = pts(i,j) - mean_(i);
        });
    }

    if(!idCov_) {
        covChol_.solveLInPlace(diff);
    }

    Kokkos::parallel_for( "log terms", N, KOKKOS_LAMBDA (const int& j) {
        output(j) = -0.5*( M*logtau_ + logDet_ );
    });

    Kokkos::parallel_for( "weighted norm", policy, KOKKOS_LAMBDA (const int& j, const int& i) {
        output(j) -= 0.5*diff(i,j)*diff(i,j);
    });
}

void GaussianDistribution<MemorySpace>::GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) {
    // Compute the gradient of the log density
    int M = pts.extent(0);
    int N = pts.extent(1);
    if(dim_ != 0 && M != dim_) {
        throw std::runtime_error("GaussianDistribution::GradLogDensityImpl: The number of rows in pts must match the dimension of the distribution.");
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{N, M}});

    if(mean_.extent(0) == 0){
        Kokkos::deep_copy(output, pts);
    }
    else {
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& j, const int& i) {
            output(i,j) = pts(i,j) - mean_(i);
        });
    }

    if(!idCov_) {
        covChol_.solveInPlace(output);
    }
}