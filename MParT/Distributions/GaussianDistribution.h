#ifndef MPART_GaussianDistribution_H
#define MPART_GaussianDistribution_H

#include "MParT/Distributions/Distribution.h"
#include "MParT/Utilities/LinearAlgebra.h"
#include <Kokkos_Core.hpp>

namespace mpart {

template<typename MemorySpace>
class GaussianDistribution: Distribution<MemorySpace> {
    public:

    GaussianDistribution(StridedVector<double, MemorySpace> mean, StridedMatrix<double, MemorySpace> covar);
    GaussianDistribution(StridedMatrix<double, MemorySpace> covar);
    GaussianDistribution(StridedVector<double, MemorySpace> mean);

    private:

#if (KOKKOS_VERSION / 10000 == 3) && (KOKKOS_VERSION / 100 % 100 < 7)
    const auto& kk_log = Kokkos::Experimental::log;
#else
    const auto& kk_log = Kokkos::log;
#endif // KOKKOS_VERSION

    static const double logtau_ = kk_log(2*Kokkos::Experimental::pi_v<double>);

    void Factorize(StridedMatrix<double, MemorySpace> Cov) {
        covLU_.compute(Cov);
        logDetCov_ = log(luSolver_.determinant());
    }

    StridedVector<double, MemorySpace> mean_;
    mpart::PartialPivLU<MemorySpace> covLU_;
    bool idCov_ = false;
    double logDetCov_;
}