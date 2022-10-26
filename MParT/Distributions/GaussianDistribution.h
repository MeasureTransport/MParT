#ifndef MPART_GaussianDistribution_H
#define MPART_GaussianDistribution_H

#include "MParT/Distributions/Distribution.h"
#include <Kokkos_Core.hpp>

namespace mpart {

template<typename MemorySpace>
class GaussianDistribution: Distribution<MemorySpace> {
    public:

    GaussianDistribution(StridedVector<double, MemorySpace> mean, StridedMatrix<double, MemorySpace> covar);
    GaussianDistribution(StridedMatrix<double, MemorySpace> covar);
    GaussianDistribution(StridedVector<double, MemorySpace> mean);

    private:

    static constexpr double tau = 2*Kokkos::Experimental::pi_v<double>;

    void Factorize(StridedMatrix<double, MemorySpace> Cov) {
        covLU_.compute(Cov);
        logDetCov_ = log(luSolver_.determinant());
    }

    StridedVector<double, MemorySpace> mean_;
    mpart::PartialPivLU<MemorySpace> covLU_;
    bool idCov_ = false;
    double logDetCov_;
}