#include "MParT/MapObjective.h"
using namespace mpart;

template<>
std::function<double(unsigned int, const double*, double*)> MapObjective<Kokkos::HostSpace>::GetOptimizationObjective(std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map) {
        StridedMatrix<const double, Kokkos::HostSpace> train = train_;
        return [this, &map, &train](unsigned int n, const double* coeffs, double* grad) {
            StridedVector<const double, Kokkos::HostSpace> coeffView = ToConstKokkos<double,Kokkos::HostSpace>(coeffs, n);
            StridedVector<double, Kokkos::HostSpace> gradView = ToKokkos<double,Kokkos::HostSpace>(grad, n);
            map->SetCoeffs(coeffView);
            return ObjectivePlusCoeffGradImpl(train, gradView, map);
        };
    }

template<typename MemorySpace>
double MapObjective<MemorySpace>::TestError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    if(test_.extent(0) == 0) {
        throw std::runtime_error("No test dataset given!");
    }
    return ObjectiveImpl(test_, map);
}

template<typename MemorySpace>
StridedVector<double, MemorySpace> MapObjective<MemorySpace>::TrainCoeffGrad(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    Kokkos::View<double, MemorySpace> grad("trainCoeffGrad", map->numCoeffs);
    CoeffGradImpl(train_, grad, map);
}

template<typename MemorySpace>
double KLObjective<MemorySpace>::ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    unsigned int N_samps = data.extent(1);
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(data);
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(data);
    double sumDensity = 0.;
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);
    ReduceColumn rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, &grad(0));
    return sumDensity/N_samps;
}

template<typename MemorySpace>
double KLObjective<MemorySpace>::ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
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
void KLObjective<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    unsigned int N_samps = data.extent(1);
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(data);
    ReduceColumn rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, &grad(0));
}

// Explicit template instantiation
template class mpart::KLObjective<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::KLObjective<DeviceSpace>;
#endif