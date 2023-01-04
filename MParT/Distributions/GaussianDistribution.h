#ifndef MPART_GaussianDistribution_H
#define MPART_GaussianDistribution_H

#include <Kokkos_Core.hpp>
#include "MParT/Distributions/Distribution.h"
#include "MParT/Utilities/LinearAlgebra.h"

namespace mpart {

template<typename MemorySpace>
class GaussianDistribution: Distribution<MemorySpace> {
    public:

    GaussianDistribution() = delete;

    /**
     * @brief Construct a new Gaussian Distribution object with custom mean and covariance
     *
     * @param mean the mean of the distribution
     * @param covar a covariance matrix for the distribution
     */
    GaussianDistribution(StridedVector<double, MemorySpace> mean, StridedMatrix<double, MemorySpace> covar);

    /**
     * @brief Construct a new Gaussian Distribution object with zero mean and custom covariance
     *
     * @param covar a covariance matrix for the distribution
     */
    GaussianDistribution(StridedMatrix<double, MemorySpace> covar);

    /**
     * @brief Construct a new Gaussian Distribution object with custom mean and identity covariance
     *
     * @param mean the mean of the distribution
     */
    GaussianDistribution(StridedVector<double, MemorySpace> mean);

    /**
     * @brief Construct a new Gaussian Distribution object representing standard Normal (zero mean, id covariance matrix)
     *
     * @param dim dimension of the distribution
     */
    GaussianDistribution(unsigned int dim);

    void SampleImpl(StridedMatrix<double, MemorySpace> output) override;
    void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) override;
    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) override;

    protected:
    using GeneratorType = typename SampleGenerator<MemorySpace>::PoolType::generator_type;
    using SampleGenerator<MemorySpace>::rand_pool;
    using SampleGenerator<MemorySpace>::dim_;

    private:

#if (KOKKOS_VERSION / 10000 < 4)

#if (KOKKOS_VERSION / 10000 == 3) && (KOKKOS_VERSION / 100 % 100 > 5)
    const double logtau_ = std::log(2*Kokkos::Experimental::pi_v<double>);
#else
    const double logtau_ = std::log(2*M_PI);
#endif

#else
    const double logtau_ = std::log(2*Kokkos::numbers::pi_v<double>);
#endif // KOKKOS_VERSION

    void Factorize(StridedMatrix<double, MemorySpace> Cov) {
        covChol_.compute(Cov);
        logDetCov_ = std::log(covChol_.determinant());
    }

    private:
    StridedVector<double, MemorySpace> mean_;
    mpart::Cholesky<MemorySpace> covChol_;
    bool idCov_ = false;
    double logDetCov_;
};

} // namespace mpart

#endif //MPART_GaussianDistribution_H