#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

template<typename MemorySpace, typename PosFuncType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_Phys_AS(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    PhysicistHermite basis1d(opts.basisNorm);
    AdaptiveSimpson<MemorySpace> quad(opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);

    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
    return output;
}

static auto reg_host_phys_as_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson), CreateComponentImpl_Phys_AS<Kokkos::HostSpace, Exp>));
static auto reg_host_phys_as_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson), CreateComponentImpl_Phys_AS<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_phys_as_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson), CreateComponentImpl_Phys_AS<mpart::DeviceSpace, Exp>));
    static auto reg_device_phys_as_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson), CreateComponentImpl_Phys_AS<mpart::DeviceSpace, SoftPlus>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_MONO_COMP(PhysicistHermite, Exp, AdaptiveSimpson, Kokkos::HostSpace)
REGISTER_MONO_COMP(PhysicistHermite, SoftPlus, AdaptiveSimpson, Kokkos::HostSpace)
#if defined(MPART_ENABLE_GPU)
REGISTER_MONO_COMP(PhysicistHermite, Exp, AdaptiveSimpson, mpart::DeviceSpace)
REGISTER_MONO_COMP(PhysicistHermite, Softplus, AdaptiveSimpson, mpart::DeviceSpace)
#endif 
#endif 