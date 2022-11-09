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

template<typename MemorySpace>
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

template<typename MemorySpace>
void GaussianDistribution<MemorySpace>::SampleImpl(StridedMatrix<double, MemorySpace> output) {
    // Sample from the distribution
    int M = output.extent(0);
    int N = output.extent(1);
    // Check dimensions
    if(dim_ != 0 && M != dim_) {
        throw std::runtime_error("GaussianDistribution::SampleImpl: The number of rows in output must match the dimension of the distribution.");
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{N, M}});

    // If the covariance is the identity, we can just sample from a shifted normal
    if(idCov_) {
        // If dim_ is 0, the distribution is the standard normal
        if(dim_ == 0) {
            Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& j, const int& i) {
                PoolType::generator_type rgen = rand_pool.get_state();
                output(i,j) = rgen.normal();
            });
        }
        // Otherwise, we sample from the mean-shifted normal
        else {
            Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& j, const int& i) {
                PoolType::generator_type rgen = rand_pool.get_state();
                output(i,j) = rgen.normal() + mean_(i);
            });
        }
    }
    // Otherwise, we assume dense covariance
    else {
        // Sample from the standard normal
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j) {
            PoolType::generator_type rgen = rand_pool.get_state();
            output(i,j) = rgen.normal();
        };
        // Transform by the Cholesky factor
        auto mul = covChol_.multiplyL(output);
        // Add the mean (if nonzero)
        if(mean_.extent(0) == 0){
            Kokkos::deep_copy(output, mul);
        }
        else {
            Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int& j, const int& i) {
                output(i,j) = mul(i,j) + mean_(i);
            });
        }
    }
}

template struct GaussianDistribution<Kokkos::HostSpace>;
#ifdef MPART_ENABLE_GPU
template struct GaussianDistribution<mpart::DeviceSpace>;
#endif