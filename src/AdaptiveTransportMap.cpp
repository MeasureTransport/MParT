#include "MParT/AdaptiveTransportMap.h"

using namespace mpart;

template<>
ATMObjective<Kokkos::HostSpace>::ATMObjective(StridedMatrix<double, Kokkos::HostSpace> x,
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map, ATMOptions options = ATMOptions()):
    x_(x), map_(map), options_(options) {
    if(options.referenceType != ReferenceTypes::StandardGaussian) {
        throw std::invalid_argument("ATMObjective<Kokkos::HostSpace>::ATMObjective: Currently only accepts Gaussian reference")
    }
}

template<>
double ATMObjective<Kokkos::HostSpace>::operator()(const std::vector<double> &coeffs, std::vector<double> &grad) {
    const unsigned int N_samps = x_.extent(1);
    StridedVector<double, Kokkos::HostSpace> coeffView = VecToKokkos(coeffs);
    StridedVector<double, Kokkos::HostSpace> gradView = VecToKokkos(grad);

    std::shared_ptr<GaussianSamplerDensity<Kokkos::HostSpace>> reference = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(x_.extent(0));
    map_->WrapCoeffs(coeffView);
    PullbackDensity<MemorySpace> pullback {map_, reference};
    StridedVector<double, Kokkos::HostSpace> densityX = pullback.LogDensity(x_);
    StridedMatrix<double, Kokkos::HostSpace> densityGradX = pullback.LogDensityCoeffGrad(x_);
    double sumDensity = 0.;
    Kokkos::parallel_reduce ("Average Negative Log Likelihood", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i)/N_samps;
    }, sumDensity);
    ReduceColumn rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, gradView.data());
}

template<>
std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> AdaptiveTransportMap(MultiIndexSet &mset0,
                StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x,
                ATMOptions options) {
    unsigned int dim = train_x.extent(0);
    MultiIndexSet mset_rm = mset.Expand();

}