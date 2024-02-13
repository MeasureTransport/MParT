#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

#include "MParT/LinearizedBasis.h"

using namespace mpart;

template<typename MemorySpace, typename PosFuncType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_LinPhys_AS(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    BasisEvaluator<BasisHomogeneity::Homogeneous,LinearizedBasis<PhysicistHermite>> basis1d(PhysicistHermite(opts.basisNorm), opts.basisLB, opts.basisUB);
    AdaptiveSimpson<MemorySpace> quad(opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv, opts.nugget);

    Kokkos::View<const double*,MemorySpace> coeffs = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
    output->SetCoeffs(coeffs);

    return output;
}

static auto reg_host_linphys_as_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, true, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson), CreateComponentImpl_LinPhys_AS<Kokkos::HostSpace, Exp>));
static auto reg_host_linphys_as_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, true, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson), CreateComponentImpl_LinPhys_AS<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_linphys_as_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, true, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson), CreateComponentImpl_LinPhys_AS<mpart::DeviceSpace, Exp>));
    static auto reg_device_linphys_as_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, true, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson), CreateComponentImpl_LinPhys_AS<mpart::DeviceSpace, SoftPlus>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, LinearizedBasis<mpart::PhysicistHermite>, Exp, AdaptiveSimpson, Kokkos::HostSpace)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, LinearizedBasis<mpart::PhysicistHermite>, SoftPlus, AdaptiveSimpson, Kokkos::HostSpace)
#if defined(MPART_ENABLE_GPU)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, LinearizedBasis<mpart::PhysicistHermite>, Exp, AdaptiveSimpson, mpart::DeviceSpace)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, LinearizedBasis<mpart::PhysicistHermite>, SoftPlus, AdaptiveSimpson, mpart::DeviceSpace)
#endif 
CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory12)
#endif
