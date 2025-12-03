#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"


using namespace mpart;


template<typename MemorySpace, typename PosFuncType, bool isCompact>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_Phys_ACC(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    BasisEvaluator<BasisHomogeneity::Homogeneous,PhysicistHermite> basis1d(opts.basisNorm);

    unsigned int level = std::log2(opts.quadPts-2);
    AdaptiveClenshawCurtis<MemorySpace> quad(level, opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

    MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,PhysicistHermite>,MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace, isCompact>>(expansion, quad, opts.contDeriv, opts.nugget);

    Kokkos::View<const double*, MemorySpace> coeffs = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
    output->SetCoeffs(coeffs);
    return output;
}

static auto reg_host_phys_acc_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis, false), CreateComponentImpl_Phys_ACC<Kokkos::HostSpace, Exp, false>));
static auto reg_host_phys_acc_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis, false), CreateComponentImpl_Phys_ACC<Kokkos::HostSpace, SoftPlus, false>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_phys_acc_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis, false), CreateComponentImpl_Phys_ACC<mpart::DeviceSpace, Exp, false>));
    static auto reg_device_phys_acc_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis, false), CreateComponentImpl_Phys_ACC<mpart::DeviceSpace, SoftPlus, false>));
#endif

static auto reg_host_phys_acc_exp_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis, true), CreateComponentImpl_Phys_ACC<Kokkos::HostSpace, Exp, true>));
static auto reg_host_phys_acc_splus_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis, true), CreateComponentImpl_Phys_ACC<Kokkos::HostSpace, SoftPlus, true>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_phys_acc_exp_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis, true), CreateComponentImpl_Phys_ACC<mpart::DeviceSpace, Exp, true>));
    static auto reg_device_phys_acc_splus_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis, true), CreateComponentImpl_Phys_ACC<mpart::DeviceSpace, SoftPlus, true>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, Exp, AdaptiveClenshawCurtis, Kokkos::HostSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, SoftPlus, AdaptiveClenshawCurtis, Kokkos::HostSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, Exp, AdaptiveClenshawCurtis, Kokkos::HostSpace, true)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, SoftPlus, AdaptiveClenshawCurtis, Kokkos::HostSpace, true)
#if defined(MPART_ENABLE_GPU)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, Exp, AdaptiveClenshawCurtis, mpart::DeviceSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, SoftPlus, AdaptiveClenshawCurtis, mpart::DeviceSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, Exp, AdaptiveClenshawCurtis, mpart::DeviceSpace, true)
REGISTER_HOMOGENEOUS_MONO_COMP(PhysicistHermite, SoftPlus, AdaptiveClenshawCurtis, mpart::DeviceSpace, true)
#endif

CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory1)
#endif 
