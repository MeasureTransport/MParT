#include "MParT/MapObjective.h"

std::function<double(unsigned int, const double*, double*)> MapObjective<Kokkos::HostSpace>::GetOptimizationObjective(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
        return [&map, &train_](unsigned int n, const double* coeffs, double* grad) {
            StridedVector<const double, MemorySpace> coeffView = ToConstKokkos<double,MemorySpace>(coeffs, n);
            StridedVector<double, MemorySpace> gradView = ToKokkos<double,MemorySpace>(grad, n);
            map->SetCoeffs(coeffView);
            return ObjectivePlusCoeffGradImpl(train_, gradView, map);
        };
    }

double MapObjective<Kokkos::HostSpace>::TestError(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    if(test_.extent(0) == 0) {
        throw std::runtime_exception("No test dataset given!");
    }
    return ObjectiveImpl(test_, map);
}

StridedVector<double, MemorySpace> MapObjective<Kokkos::HostSpace>::TrainCoeffGrad(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    Kokkos::View<double, MemorySpace> grad("trainCoeffGrad", map->numCoeffs);
    CoeffGradImpl(train_, grad, map);
}

double KLObjective<Kokkos::HostSpace>::ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(x_);
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(x_);
    double sumDensity = 0.;
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);
    ReduceColumn rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, &grad(0));
    return sumDensity/N_samps;
}

double KLObjective<Kokkos::HostSpace>::ObjectiveImpl(StridedMatrix<const double, MemorySpace> data, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(x_);
    double sumDensity = 0.;
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);
    return sumDensity/N_samps;
}

void KLObjective<Kokkos::HostSpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(x_);
    ReduceColumn rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, &grad(0));
}