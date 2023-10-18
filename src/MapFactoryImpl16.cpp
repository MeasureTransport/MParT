#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/HermiteFunction.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

#include "MParT/LinearizedBasis.h"

using namespace mpart;

template<typename MemorySpace, typename PosFuncType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_LinHF_AS(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    LinearizedBasis<HermiteFunction> basis1d(HermiteFunction(), opts.basisLB, opts.basisUB);
    AdaptiveSimpson<MemorySpace> quad(opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);

    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
    return output;
}

static auto reg_host_linhf_as_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, true, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson), CreateComponentImpl_LinHF_AS<Kokkos::HostSpace, Exp>));
static auto reg_host_linhf_as_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, true, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson), CreateComponentImpl_LinHF_AS<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_linhf_as_exp = mpart::MapFactory::CompFactoryImpl<DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, true, PosFuncTypes::Exp, QuadTypes::AdaptiveSimpson), CreateComponentImpl_LinHF_AS<Kokkos::HostSpace, Exp>));
    static auto reg_device_linhf_as_splus = mpart::MapFactory::CompFactoryImpl<DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::HermiteFunctions, true, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveSimpson), CreateComponentImpl_LinHF_AS<Kokkos::HostSpace, SoftPlus>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_MONO_COMP(LinearizedBasis<mpart::HermiteFunction>, Exp, AdaptiveSimpson, Kokkos::HostSpace)
REGISTER_MONO_COMP(LinearizedBasis<mpart::HermiteFunction>, SoftPlus, AdaptiveSimpson, Kokkos::HostSpace)
#if defined(MPART_ENABLE_GPU)
REGISTER_MONO_COMP(LinearizedBasis<mpart::HermiteFunction>, Exp, AdaptiveSimpson, mpart::DeviceSpace)
REGISTER_MONO_COMP(LinearizedBasis<mpart::HermiteFunction>, Softplus, AdaptiveSimpson, mpart::DeviceSpace)
#endif 
CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory16)
#endif 