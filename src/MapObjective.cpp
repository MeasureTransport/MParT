#include "MParT/MapObjective.h"
using namespace mpart;

template <typename MemorySpace>
double MapObjective<MemorySpace>::operator()(
    unsigned int n, const double* coeffs, double* grad,
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
  StridedVector<const double, MemorySpace> coeffView =
      ToConstKokkos<double, MemorySpace>(coeffs, n);
  StridedVector<double, MemorySpace> gradView =
      ToKokkos<double, MemorySpace>(grad, n);
  map->SetCoeffs(coeffView);
  return ObjectivePlusCoeffGradImpl(train_, gradView, map);
}

template <typename MemorySpace>
double MapObjective<MemorySpace>::TestError(
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
  if (test_.extent(0) == 0) {
    throw std::runtime_error("No test dataset given!");
  }
  return ObjectiveImpl(test_, map);
}

template <typename MemorySpace>
double MapObjective<MemorySpace>::TrainError(
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
  return ObjectiveImpl(train_, map);
}

template <typename MemorySpace>
void MapObjective<MemorySpace>::TrainCoeffGradImpl(
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map,
    StridedVector<double, MemorySpace> grad) const {
  CoeffGradImpl(train_, grad, map);
}

template <typename MemorySpace>
StridedVector<double, MemorySpace> MapObjective<MemorySpace>::TrainCoeffGrad(
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
  Kokkos::View<double*, MemorySpace> grad("trainCoeffGrad", map->numCoeffs);
  TrainCoeffGradImpl(map, grad);
  return grad;
}

template <typename MemorySpace>
std::shared_ptr<MapObjective<MemorySpace>>
ObjectiveFactory::CreateGaussianKLObjective(
    StridedMatrix<const double, MemorySpace> train, unsigned int dim) {
  if (dim == 0) dim = train.extent(0);
  std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density =
      std::make_shared<GaussianSamplerDensity<MemorySpace>>(dim);
  return std::make_shared<KLObjective<MemorySpace>>(train, density);
}

template <typename MemorySpace>
std::shared_ptr<MapObjective<MemorySpace>>
ObjectiveFactory::CreateGaussianKLObjective(
    StridedMatrix<const double, MemorySpace> train,
    StridedMatrix<const double, MemorySpace> test, unsigned int dim) {
  if (dim == 0) dim = train.extent(0);
  std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density =
      std::make_shared<GaussianSamplerDensity<MemorySpace>>(dim);
  return std::make_shared<KLObjective<MemorySpace>>(train, test, density);
}

template <typename MemorySpace>
double KLObjective<MemorySpace>::ObjectivePlusCoeffGradImpl(
    StridedMatrix<const double, MemorySpace> data,
    StridedVector<double, MemorySpace> grad,
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
  unsigned int N_samps = data.extent(1);
  PullbackDensity<MemorySpace> pullback{map, density_};
  StridedVector<double, MemorySpace> densityX = pullback.LogDensity(data);
  StridedMatrix<double, MemorySpace> densityGradX =
      pullback.LogDensityCoeffGrad(data);
  double sumDensity = 0.;
  Kokkos::parallel_reduce(
      "Sum Negative Log Likelihood", N_samps,
      KOKKOS_LAMBDA(const int i, double& sum) { sum -= densityX(i); },
      sumDensity);
  ReduceDim<ReduceDimMap::sum, MemorySpace> rc(densityGradX,
                                               -1.0 / ((double)N_samps));
  Kokkos::parallel_reduce(N_samps, rc, &grad(0));
  return sumDensity / N_samps;
}

template <typename MemorySpace>
double KLObjective<MemorySpace>::ObjectiveImpl(
    StridedMatrix<const double, MemorySpace> data,
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
  unsigned int N_samps = data.extent(1);
  PullbackDensity<MemorySpace> pullback{map, density_};
  StridedVector<double, MemorySpace> densityX = pullback.LogDensity(data);
  double sumDensity = 0.;
  Kokkos::parallel_reduce(
      "Sum Negative Log Likelihood", N_samps,
      KOKKOS_LAMBDA(const int i, double& sum) { sum -= densityX(i); },
      sumDensity);
  return sumDensity / N_samps;
}

template <typename MemorySpace>
void KLObjective<MemorySpace>::CoeffGradImpl(
    StridedMatrix<const double, MemorySpace> data,
    StridedVector<double, MemorySpace> grad,
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
  unsigned int N_samps = data.extent(1);
  PullbackDensity<MemorySpace> pullback{map, density_};
  StridedMatrix<double, MemorySpace> densityGradX =
      pullback.LogDensityCoeffGrad(data);
  ReduceDim<ReduceDimMap::sum, MemorySpace> rc(densityGradX,
                                               -1.0 / ((double)N_samps));
  Kokkos::parallel_reduce(N_samps, rc, &grad(0));
}

template <typename MemorySpace>
double ParamL2RegularizationObjective<MemorySpace>::ObjectivePlusCoeffGradImpl(
    StridedMatrix<const double, MemorySpace>,
    StridedVector<double, MemorySpace> grad,
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
  double sq_norm = 0;
  const double scale = this->scale_;
  const StridedVector<const double, MemorySpace> coeffs = map->Coeffs();
  Kokkos::parallel_reduce(
      "Square L2 Norm of Params", map->numCoeffs,
      KOKKOS_LAMBDA(const int& i, double& lnorm) {
        lnorm += coeffs(i) * coeffs(i);
        grad(i) += 2 * scale * coeffs(i);
      },
      sq_norm);
  return scale * sq_norm;
}

template <typename MemorySpace>
double ParamL2RegularizationObjective<MemorySpace>::ObjectiveImpl(
    StridedMatrix<const double, MemorySpace>,
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
  double sq_norm = 0;
  StridedVector<const double, MemorySpace> coeffs = map->Coeffs();
  Kokkos::parallel_reduce(
      "Square L2 Norm of Params", map->numCoeffs,
      KOKKOS_LAMBDA(const int& i, double& lnorm) {
        lnorm += coeffs(i) * coeffs(i);
      },
      sq_norm);
  return sq_norm * this->scale_;
}

template <typename MemorySpace>
void ParamL2RegularizationObjective<MemorySpace>::CoeffGradImpl(
    StridedMatrix<const double, MemorySpace>,
    StridedVector<double, MemorySpace> grad,
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map) const {
  double scale = this->scale_;
  Kokkos::parallel_for(
      "CoeffGrad of Regularization", map->numCoeffs,
      KOKKOS_LAMBDA(const int i) { grad(i) += 2 * scale * map->Coeffs()(i); });
}



// Explicit template instantiation
template class mpart::MapObjective<Kokkos::HostSpace>;
template class mpart::KLObjective<Kokkos::HostSpace>;
template class mpart::ParamL2RegularizationObjective<Kokkos::HostSpace>;
template std::shared_ptr<MapObjective<Kokkos::HostSpace>>
mpart::ObjectiveFactory::CreateGaussianKLObjective<Kokkos::HostSpace>(
    StridedMatrix<const double, Kokkos::HostSpace>, unsigned int);
template std::shared_ptr<MapObjective<Kokkos::HostSpace>>
mpart::ObjectiveFactory::CreateGaussianKLObjective<Kokkos::HostSpace>(
    StridedMatrix<const double, Kokkos::HostSpace>,
    StridedMatrix<const double, Kokkos::HostSpace>, unsigned int);
#if defined(MPART_ENABLE_GPU)
template class mpart::MapObjective<DeviceSpace>;
template class mpart::KLObjective<DeviceSpace>;
template std::shared_ptr<MapObjective<DeviceSpace>>
mpart::ObjectiveFactory::CreateGaussianKLObjective<DeviceSpace>(
    StridedMatrix<const double, DeviceSpace>, unsigned int);
template std::shared_ptr<MapObjective<DeviceSpace>>
mpart::ObjectiveFactory::CreateGaussianKLObjective<DeviceSpace>(
    StridedMatrix<const double, DeviceSpace>,
    StridedMatrix<const double, DeviceSpace>, unsigned int);
#endif