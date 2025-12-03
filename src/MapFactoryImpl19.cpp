#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/UnivariateBases.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

template<typename MemorySpace, typename PosFuncType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_LEG_ACC(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    BasisEvaluator<BasisHomogeneity::Homogeneous,ShiftedLegendre,Identity> basis1d(mset.Length());
    unsigned int level = std::log2(opts.quadPts-2);
    AdaptiveClenshawCurtis<MemorySpace> quad(level, opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace, true>>(expansion, quad, opts.contDeriv, opts.nugget);

    Kokkos::View<const double*,MemorySpace> coeffs = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
    output->SetCoeffs(coeffs);

    return output;
}

static auto reg_host_leg_acc_exp_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::Legendre, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis, true), CreateComponentImpl_LEG_ACC<Kokkos::HostSpace, Exp>));
static auto reg_host_leg_acc_splus_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::Legendre, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis, true), CreateComponentImpl_LEG_ACC<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_leg_acc_exp_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::Legendre, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis, true), CreateComponentImpl_LEG_ACC<mpart::DeviceSpace, Exp>));
    static auto reg_device_leg_acc_splus_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::Legendre, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis, true), CreateComponentImpl_LEG_ACC<mpart::DeviceSpace, SoftPlus>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_HOMOGENEOUS_MONO_COMP(ShiftedLegendre, Exp, AdaptiveClenshawCurtis, Kokkos::HostSpace, true)
REGISTER_HOMOGENEOUS_MONO_COMP(ShiftedLegendre, SoftPlus, AdaptiveClenshawCurtis, Kokkos::HostSpace, true)
#if defined(MPART_ENABLE_GPU)
REGISTER_HOMOGENEOUS_MONO_COMP(ShiftedLegendre, Exp, AdaptiveClenshawCurtis, mpart::DeviceSpace, true)
REGISTER_HOMOGENEOUS_MONO_COMP(ShiftedLegendre, SoftPlus, AdaptiveClenshawCurtis, mpart::DeviceSpace, true)
#endif
CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory19)
#endif