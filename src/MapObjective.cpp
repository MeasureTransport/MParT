#include "MParT/MapObjective.h"
using namespace mpart;

template<typename MemorySpace>
double MapObjective<MemorySpace>::operator()(unsigned int n, const double* coeffs, double* grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {

    StridedVector<const double, MemorySpace> coeffView = ToConstKokkos<double,MemorySpace>(coeffs, n);
    StridedVector<double, MemorySpace> gradView = ToKokkos<double,MemorySpace>(grad, n);
    map->SetCoeffs(coeffView);
    return ObjectivePlusCoeffGradImpl(train_, gradView, map);
}

template<typename MemorySpace>
double MapObjective<MemorySpace>::TestError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    if(test_.extent(0) == 0) {
        throw std::runtime_error("No test dataset given!");
    }
    return ObjectiveImpl(test_, map);
}

template<typename MemorySpace>
double MapObjective<MemorySpace>::TrainError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    return ObjectiveImpl(train_, map);
}

template<typename MemorySpace>
StridedVector<double, MemorySpace> MapObjective<MemorySpace>::TrainCoeffGrad(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    Kokkos::View<double*, MemorySpace> grad("trainCoeffGrad", map->numCoeffs);
    CoeffGradImpl(train_, grad, map);
    return grad;
}

template<typename MemorySpace>
double KLObjective<MemorySpace>::ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    unsigned int N_samps = data.extent(1);
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(data);
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(data);
    double sumDensity = 0.;
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);
    ReduceDim<ReduceDimMap::sum,MemorySpace> rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, &grad(0));
    Kokkos::fence("End of MapObjective");
    return sumDensity/N_samps;
}

template<typename MemorySpace>
double KLObjective<MemorySpace>::ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    unsigned int N_samps = data.extent(1);
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(data);
    double sumDensity = 0.;
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);
    return sumDensity/N_samps;
}

template<typename MemorySpace>
void KLObjective<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    unsigned int N_samps = data.extent(1);
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(data);
    ReduceDim<ReduceDimMap::sum,MemorySpace> rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, &grad(0));
}

// Explicit template instantiation
template class mpart::MapObjective<Kokkos::HostSpace>;
template class mpart::KLObjective<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::MapObjective<DeviceSpace>;
    template class mpart::KLObjective<DeviceSpace>;
#endif