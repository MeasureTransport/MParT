#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/HermiteFunction.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

template<typename MemorySpace, typename PosFuncType, bool isCompact>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_HF_CC(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    BasisEvaluator<BasisHomogeneity::Homogeneous,HermiteFunction> basis1d;
    ClenshawCurtisQuadrature<MemorySpace> quad(opts.quadPts, 1);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace, isCompact>>(expansion, quad, opts.contDeriv, opts.nugget);

    Kokkos::View<const double*,MemorySpace> coeffs = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
    output->SetCoeffs(coeffs);

    return output;
}

static auto reg_host_hf_cc_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis, false), CreateComponentImpl_HF_CC<Kokkos::HostSpace, Exp, false>));
static auto reg_host_hf_cc_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis, false), CreateComponentImpl_HF_CC<Kokkos::HostSpace, SoftPlus, false>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_hf_cc_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis, false), CreateComponentImpl_HF_CC<mpart::DeviceSpace, Exp, false>));
    static auto reg_device_hf_cc_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis, false), CreateComponentImpl_HF_CC<mpart::DeviceSpace, SoftPlus, false>));
#endif

static auto reg_host_hf_cc_exp_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis, true), CreateComponentImpl_HF_CC<Kokkos::HostSpace, Exp, true>));
static auto reg_host_hf_cc_splus_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis, true), CreateComponentImpl_HF_CC<Kokkos::HostSpace, SoftPlus, true>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_hf_cc_exp_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis, true), CreateComponentImpl_HF_CC<mpart::DeviceSpace, Exp, true>));
    static auto reg_device_hf_cc_splus_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis, true), CreateComponentImpl_HF_CC<mpart::DeviceSpace, SoftPlus, true>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_HOMOGENEOUS_MONO_COMP(HermiteFunction, Exp, ClenshawCurtisQuadrature, Kokkos::HostSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(HermiteFunction, SoftPlus, ClenshawCurtisQuadrature, Kokkos::HostSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(HermiteFunction, Exp, ClenshawCurtisQuadrature, Kokkos::HostSpace, true)
REGISTER_HOMOGENEOUS_MONO_COMP(HermiteFunction, SoftPlus, ClenshawCurtisQuadrature, Kokkos::HostSpace, true)
#if defined(MPART_ENABLE_GPU)
REGISTER_HOMOGENEOUS_MONO_COMP(HermiteFunction, Exp, ClenshawCurtisQuadrature, mpart::DeviceSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(HermiteFunction, SoftPlus, ClenshawCurtisQuadrature, mpart::DeviceSpace, false)
REGISTER_HOMOGENEOUS_MONO_COMP(HermiteFunction, Exp, ClenshawCurtisQuadrature, mpart::DeviceSpace, true)
REGISTER_HOMOGENEOUS_MONO_COMP(HermiteFunction, SoftPlus, ClenshawCurtisQuadrature, mpart::DeviceSpace, true)
#endif
CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory8)
#endif