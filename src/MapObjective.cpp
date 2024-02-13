#include "MParT/MapObjective.h"

using namespace mpart;

template<typename MemorySpace>
double MapObjective<MemorySpace>::operator()(unsigned int n, const double* coeffs, double* grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {

    Kokkos::View<const double*, MemorySpace> coeffView = ToConstKokkos<double,MemorySpace>(coeffs, n);
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
void MapObjective<MemorySpace>::TrainCoeffGradImpl(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, StridedVector<double, MemorySpace> grad) const {
    CoeffGradImpl(train_, grad, map);
}

template<typename MemorySpace>
StridedVector<double, MemorySpace> MapObjective<MemorySpace>::TrainCoeffGrad(std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    Kokkos::View<double*, MemorySpace> grad("trainCoeffGrad", map->numCoeffs);
    TrainCoeffGradImpl(map, grad);
    return grad;
}

template<typename MemorySpace>
std::shared_ptr<MapObjective<MemorySpace>> ObjectiveFactory::CreateGaussianKLObjective(StridedMatrix<const double, MemorySpace> train, unsigned int dim) {
    if(dim == 0) dim = train.extent(0);
    std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density = std::make_shared<GaussianSamplerDensity<MemorySpace>>(dim);
    return std::make_shared<KLObjective<MemorySpace>>(train, density);
}

template<typename MemorySpace>
std::shared_ptr<MapObjective<MemorySpace>> ObjectiveFactory::CreateGaussianKLObjective(StridedMatrix<const double, MemorySpace> train, StridedMatrix<const double, MemorySpace> test, unsigned int dim) {
    if(dim == 0) dim = train.extent(0);
    std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density = std::make_shared<GaussianSamplerDensity<MemorySpace>>(dim);
    return std::make_shared<KLObjective<MemorySpace>>(train, test, density);
}

template<typename MemorySpace>
double KLObjective<MemorySpace>::ObjectivePlusCoeffGradImpl(StridedMatrix<const double, MemorySpace> data, StridedVector<double, MemorySpace> grad, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
    using ExecSpace = typename MemoryToExecution<MemorySpace>::Space;
    unsigned int N_samps = data.extent(1);
    unsigned int grad_dim = grad.extent(0);
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(data);
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(data);
    double sumDensity = 0.;
    
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);


    if (grad.data()!=nullptr){
        double scale = -1.0/((double) N_samps);

        Kokkos::TeamPolicy<ExecSpace> policy(grad_dim, Kokkos::AUTO());
        using team_handle = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
        Kokkos::parallel_for(policy,
            KOKKOS_LAMBDA(const team_handle& teamMember){
                int row = teamMember.league_rank();
                double thisRowSum = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N_samps), 
                [&] (const int& col, double& innerUpdate) {
                    innerUpdate += scale*densityGradX(row,col);
                },
                thisRowSum);

                grad(row) = thisRowSum;
            }
        );
        Kokkos::fence();
    }
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
    using ExecSpace = typename MemoryToExecution<MemorySpace>::Space;
    unsigned int N_samps = data.extent(1);
    unsigned int grad_dim = grad.extent(0);
    PullbackDensity<MemorySpace> pullback {map, density_};
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(data);

    double scale = -1.0/((double) N_samps);
    Kokkos::TeamPolicy<ExecSpace> policy(grad_dim, Kokkos::AUTO());
    using team_handle = typename Kokkos::TeamPolicy<ExecSpace>::member_type;
    Kokkos::parallel_for(policy,
        KOKKOS_LAMBDA(const team_handle& teamMember){
            int row = teamMember.league_rank();
            double thisRowSum = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(teamMember, N_samps), 
            [&] (const int& col, double& innerUpdate) {
                innerUpdate += scale*densityGradX(row,col);
            },
            thisRowSum);
            grad(row) = thisRowSum;
        }
    );
    Kokkos::fence();
}

// Explicit template instantiation
template class mpart::MapObjective<Kokkos::HostSpace>;
template class mpart::KLObjective<Kokkos::HostSpace>;
template std::shared_ptr<MapObjective<Kokkos::HostSpace>> mpart::ObjectiveFactory::CreateGaussianKLObjective<Kokkos::HostSpace>(StridedMatrix<const double, Kokkos::HostSpace>, unsigned int);
template std::shared_ptr<MapObjective<Kokkos::HostSpace>> mpart::ObjectiveFactory::CreateGaussianKLObjective<Kokkos::HostSpace>(StridedMatrix<const double, Kokkos::HostSpace>, StridedMatrix<const double, Kokkos::HostSpace>, unsigned int);
#if defined(MPART_ENABLE_GPU)
    template class mpart::MapObjective<DeviceSpace>;
    template class mpart::KLObjective<DeviceSpace>;
    template std::shared_ptr<MapObjective<DeviceSpace>> mpart::ObjectiveFactory::CreateGaussianKLObjective<DeviceSpace>(StridedMatrix<const double, DeviceSpace>, unsigned int);
    template std::shared_ptr<MapObjective<DeviceSpace>> mpart::ObjectiveFactory::CreateGaussianKLObjective<DeviceSpace>(StridedMatrix<const double, DeviceSpace>, StridedMatrix<const double, DeviceSpace>, unsigned int);
#endif
