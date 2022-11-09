#ifndef MPART_GaussianDistribution_H
#define MPART_GaussianDistribution_H

#include <Kokkos_Core.hpp>
#include "MParT/Distributions/Distribution.h"
#include "MParT/Utilities/LinearAlgebra.h"

namespace mpart {

template<typename MemorySpace>
class GaussianDistribution: Distribution<MemorySpace> {
    public:

    GaussianDistribution(StridedVector<double, MemorySpace> mean, StridedMatrix<double, MemorySpace> covar);
    GaussianDistribution(StridedMatrix<double, MemorySpace> covar);
    GaussianDistribution(StridedVector<double, MemorySpace> mean);
    GaussianDistribution();

    private:

#if (KOKKOS_VERSION / 10000 == 3) && (KOKKOS_VERSION / 100 % 100 < 7)
    const auto& kk_log = Kokkos::Experimental::log;
#else
    const auto& kk_log = Kokkos::log;
#endif // KOKKOS_VERSION

    static const double logtau_ = kk_log(2*Kokkos::Experimental::pi_v<double>);

    void Factorize(StridedMatrix<double, MemorySpace> Cov) {
        covChol_.compute(Cov);
        logDetCov_ = log(covChol_.determinant());
    }

    StridedVector<double, MemorySpace> mean_;
    mpart::Cholesky<MemorySpace> covChol_;
    bool idCov_ = false;
    unsigned int dim_ = 0;
    double logDetCov_;
}