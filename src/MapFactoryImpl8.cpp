#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/HermiteFunction.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

template<typename MemorySpace, typename PosFuncType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_HF_CC(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    BasisEvaluator<BasisHomogeneity::Homogeneous,HermiteFunction> basis1d;
    ClenshawCurtisQuadrature<MemorySpace> quad(opts.quadPts, 1);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv, opts.nugget);

    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
    return output;
}

static auto reg_host_hf_cc_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis), CreateComponentImpl_HF_CC<Kokkos::HostSpace, Exp>));
static auto reg_host_hf_cc_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis), CreateComponentImpl_HF_CC<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_hf_cc_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis), CreateComponentImpl_HF_CC<mpart::DeviceSpace, Exp>));
    static auto reg_device_hf_cc_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis), CreateComponentImpl_HF_CC<mpart::DeviceSpace, SoftPlus>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, HermiteFunction, Exp, ClenshawCurtisQuadrature, Kokkos::HostSpace)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, HermiteFunction, SoftPlus, ClenshawCurtisQuadrature, Kokkos::HostSpace)
#if defined(MPART_ENABLE_GPU)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, HermiteFunction, Exp, ClenshawCurtisQuadrature, mpart::DeviceSpace)
REGISTER_MONO_COMP(BasisHomogeneity::Homogeneous, HermiteFunction, Softplus, ClenshawCurtisQuadrature, mpart::DeviceSpace)
#endif 
CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory8)
#endif