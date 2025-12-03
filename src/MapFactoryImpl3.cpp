#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

template<typename MemorySpace, typename PosFuncType, bool isCompact>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_Phys_AS(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    BasisEvaluator<BasisHomogeneity::Homogeneous,PhysicistHermite> basis1d(opts.basisNorm);
    AdaptiveSimpson<MemorySpace> quad(opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace, isCompact>>(expansion, quad, opts.contDeriv, opts.nugget);

    Kokkos::View<const double*,MemorySpace> coeffs = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
    output->SetCoeffs(coeffs);

    return output;
}

static auto reg_host_phys_as_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson, false), CreateComponentImpl_Phys_AS<Kokkos::HostSpace, Exp, false>));
static auto reg_host_phys_as_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson, false), CreateComponentImpl_Phys_AS<Kokkos::HostSpace, SoftPlus, false>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_phys_as_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson, false), CreateComponentImpl_Phys_AS<mpart::DeviceSpace, Exp, false>));
    static auto reg_device_phys_as_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson, false), CreateComponentImpl_Phys_AS<mpart::DeviceSpace, SoftPlus, false>));
#endif

static auto reg_host_phys_as_exp_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson, true), CreateComponentImpl_Phys_AS<Kokkos::HostSpace, Exp, true>));
static auto reg_host_phys_as_splus_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson, true), CreateComponentImpl_Phys_AS<Kokkos::HostSpace, SoftPlus, true>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_phys_as_exp_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson, true), CreateComponentImpl_Phys_AS<mpart::DeviceSpace, Exp, true>));
    static auto reg_device_phys_as_splus_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson, true), CreateComponentImpl_Phys_AS<mpart::DeviceSpace, SoftPlus, true>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, Exp, AdaptiveSimpson, Kokkos::HostSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, SoftPlus, AdaptiveSimpson, Kokkos::HostSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, Exp, AdaptiveSimpson, Kokkos::HostSpace, true)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, SoftPlus, AdaptiveSimpson, Kokkos::HostSpace, true)
#if defined(MPART_ENABLE_GPU)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, Exp, AdaptiveSimpson, mpart::DeviceSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, SoftPlus, AdaptiveSimpson, mpart::DeviceSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, Exp, AdaptiveSimpson, mpart::DeviceSpace, true)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, SoftPlus, AdaptiveSimpson, mpart::DeviceSpace, true)
#endif 
CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory3)
#endif 
