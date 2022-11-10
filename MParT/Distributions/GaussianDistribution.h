#ifndef MPART_GaussianDistribution_H
#define MPART_GaussianDistribution_H

#include <Kokkos_Core.hpp>
#include "MParT/Distributions/Distribution.h"
#include "MParT/Utilities/LinearAlgebra.h"

namespace mpart {

template<typename MemorySpace>
class GaussianSamplerDensity {
    public:

    GaussianSamplerDensity() = delete;
    GaussianSamplerDensity(StridedVector<double, MemorySpace> mean, StridedMatrix<double, MemorySpace> covar);
    GaussianSamplerDensity(StridedMatrix<double, MemorySpace> covar);
    GaussianSamplerDensity(StridedVector<double, MemorySpace> mean);
    GaussianSamplerDensity(unsigned int dim);

    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> &output);
    void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output);
    void SampleImpl(StridedMatrix<double, MemorySpace> output);

    private:

#if (KOKKOS_VERSION / 10000 == 3) && (KOKKOS_VERSION / 100 % 100 < 7)
    const double logtau_ = std::log(2*Kokkos::Experimental::pi_v<double>);
#else
    const double logtau_ = std::log(2*Kokkos::numbers::pi_v<double>);
#endif // KOKKOS_VERSION

    void Factorize(StridedMatrix<double, MemorySpace> Cov) {
        covChol_.compute(Cov);
        logDetCov_ = std::log(covChol_.determinant());
    }

    void SetSeed(unsigned int seed) {
        rand_pool = PoolType(seed);
    }

    using PoolType = Kokkos::Random_XorShift64_Pool<typename MemoryToExecution<MemorySpace>::Space>;

    private:
    PoolType rand_pool;
    StridedVector<double, MemorySpace> mean_;
    mpart::Cholesky<MemorySpace> covChol_;
    bool idCov_ = false;
    unsigned int dim_ = 0;
    double logDetCov_;
};

template<typename MemorySpace>
using GaussianDistribution = Distribution<MemorySpace, GaussianSamplerDensity<MemorySpace>, GaussianSamplerDensity<MemorySpace>>;

} // namespace mpart

#endif //MPART_GaussianDistribution_H