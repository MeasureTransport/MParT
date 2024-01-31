#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/HermiteFunction.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

template<typename MemorySpace, typename PosFuncType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_HF_ACC(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    BasisEvaluator<BasisHomogeneity::Homogeneous,HermiteFunction> basis1d;

    unsigned int level = std::log2(opts.quadPts-2);
    AdaptiveClenshawCurtis<MemorySpace> quad(level, opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv, opts.nugget);

    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
    return output;
}

static auto reg_host_hf_acc_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis), CreateComponentImpl_HF_ACC<Kokkos::HostSpace, Exp>));
static auto reg_host_hf_acc_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis), CreateComponentImpl_HF_ACC<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_hf_acc_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis), CreateComponentImpl_HF_ACC<mpart::DeviceSpace, Exp>));
    static auto reg_device_hf_acc_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis), CreateComponentImpl_HF_ACC<mpart::DeviceSpace, SoftPlus>));
#endif


#if defined(MPART_HAS_CEREAL)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, HermiteFunction, Exp, AdaptiveClenshawCurtis, Kokkos::HostSpace)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, HermiteFunction, SoftPlus, AdaptiveClenshawCurtis, Kokkos::HostSpace)
#if defined(MPART_ENABLE_GPU)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, HermiteFunction, Exp, AdaptiveClenshawCurtis, mpart::DeviceSpace)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, HermiteFunction, Softplus, AdaptiveClenshawCurtis, mpart::DeviceSpace)
#endif
CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory7) 
#endif