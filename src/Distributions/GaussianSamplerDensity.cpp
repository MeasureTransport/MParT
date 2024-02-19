#include "MParT/Distributions/GaussianSamplerDensity.h"

using namespace mpart;

template<typename MemorySpace>
GaussianSamplerDensity<MemorySpace>::GaussianSamplerDensity(StridedVector<double, MemorySpace> mean, StridedMatrix<double, MemorySpace> covar): SampleGenerator<MemorySpace>(mean.extent(0)), DensityBase<MemorySpace>(mean.extent(0)), mean_(mean) {
    Factorize(covar);
}

template<typename MemorySpace>
GaussianSamplerDensity<MemorySpace>::GaussianSamplerDensity(StridedMatrix<double, MemorySpace> covar): SampleGenerator<MemorySpace>(covar.extent(0)), DensityBase<MemorySpace>(covar.extent(0)) {
    Factorize(covar);
}

template<typename MemorySpace>
GaussianSamplerDensity<MemorySpace>::GaussianSamplerDensity(StridedVector<double, MemorySpace> mean): SampleGenerator<MemorySpace>(mean.extent(0)), DensityBase<MemorySpace>(mean.extent(0)), mean_(mean), idCov_(true) {}

template<typename MemorySpace>
GaussianSamplerDensity<MemorySpace>::GaussianSamplerDensity(unsigned int dim): SampleGenerator<MemorySpace>(dim), DensityBase<MemorySpace>(dim), idCov_(true) {}

template<typename MemorySpace>
void GaussianSamplerDensity<MemorySpace>::LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) {
    // Compute the log density
    int M = pts.extent(0);
    int N = pts.extent(1);
    if(M != dim_) {
        throw std::runtime_error("GaussianSamplerDensity::LogDensityImpl: The number of rows in pts must match the dimension of the distribution.");
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{N, M}});
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> diff ("diff", M, N);

    if(mean_.extent(0) == 0){
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& j, const int& i) {
            diff(i,j) = pts(i,j);
        });
    }
    else {
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& j, const int& i) {
            diff(i,j) = pts(i,j) - mean_(i);
        });
    }

    if(!idCov_) {
        covChol_.solveInPlaceL(diff);
    }

    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy1d{0, N};
    Kokkos::parallel_for(policy1d, KOKKOS_CLASS_LAMBDA(const int& j){
        output(j) = -0.5*( M*logtau_ + logDetCov_ );
        for(int d=0; d<M; ++d){
            output(j) += -0.5*diff(d,j)*diff(d,j);
        }
    });
}

template<typename MemorySpace>
void GaussianSamplerDensity<MemorySpace>::LogDensityInputGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) {
    // Compute the gradient of the log density
    int M = pts.extent(0);
    int N = pts.extent(1);
    if(M != dim_) {
        throw std::runtime_error("GaussianSamplerDensity::LogDensityInputGradImpl: The number of rows in pts must match the dimension of the distribution.");
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{N, M}});

    if(mean_.extent(0) == 0){
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& j, const int& i) {
            output(i,j) = -pts(i,j);
        });
    }
    else {
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int& j, const int& i) {
            output(i,j) = -pts(i,j) + mean_(i);
        });
    }

    if(!idCov_) {
        covChol_.solveInPlace(output);
    }
}


// Currently this requires that output be a LayoutLeft view
template<typename MemorySpace>
void GaussianSamplerDensity<MemorySpace>::SampleImpl(StridedMatrix<double, MemorySpace> output_) {
    // Sample from the distribution
    int M = output_.extent(0);
    int N = output_.extent(1);
    // Check dimensions
    if(M != dim_) {
        throw std::runtime_error("GaussianSamplerDensity::SampleImpl: The number of rows in output must match the dimension of the distribution.");
    }
    Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({{0, 0}}, {{N, M}});
    // If the covariance is the identity, we can just sample from a shifted normal
    if(idCov_) {
        // If dim_ is 0, the distribution is the standard normal
        if(mean_.extent(0) == 0) {
            Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int j, const int i) {
                GeneratorType rgen = rand_pool.get_state();
                output_(i,j) = rgen.normal();
                rand_pool.free_state(rgen);
            });
        }
        // Otherwise, we sample from the mean-shifted normal
        else {
            Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int j, const int i) {
                GeneratorType rgen = rand_pool.get_state();
                output_(i,j) = rgen.normal() + mean_(i);
                rand_pool.free_state(rgen);
            });
        }
    }
    // Otherwise, we assume dense covariance
    else {
        // Enforce that the output is the correct layout!
        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> output = output_;
        // Sample from the standard normal
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int j, const int i) {
            GeneratorType rgen = rand_pool.get_state();
            output(i,j) = rgen.normal();
            rand_pool.free_state(rgen);
        });
        // Transform by the Cholesky factor
        auto mul = covChol_.multiplyL(output);
        // Add the mean (if nonzero)
        if(mean_.extent(0) == 0){
            Kokkos::deep_copy(output, mul);
        }
        else {
            Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(const int j, const int i) {
                output(i,j) = mul(i,j) + mean_(i);
            });
        }
    }
}

template struct mpart::GaussianSamplerDensity<Kokkos::HostSpace>;
#ifdef MPART_ENABLE_GPU
template struct mpart::GaussianSamplerDensity<mpart::DeviceSpace>;
#endif
