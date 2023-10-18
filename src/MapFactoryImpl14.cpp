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
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_LinProb_CC(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    LinearizedBasis<ProbabilistHermite> basis1d(ProbabilistHermite(opts.basisNorm), opts.basisLB, opts.basisUB);
    ClenshawCurtisQuadrature<MemorySpace> quad(opts.quadPts, 1);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);

    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
    return output;
}

static auto reg_host_linprob_cc_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::ProbabilistHermite, true, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis), CreateComponentImpl_LinProb_CC<Kokkos::HostSpace, Exp>));
static auto reg_host_linprob_cc_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::ProbabilistHermite, true, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis), CreateComponentImpl_LinProb_CC<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_linprob_cc_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::ProbabilistHermite, true, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis), CreateComponentImpl_LinProb_CC<mpart::DeviceSpace, Exp>));
    static auto reg_device_linprob_cc_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::ProbabilistHermite, true, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis), CreateComponentImpl_LinProb_CC<mpart::DeviceSpace, SoftPlus>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_MONO_COMP(LinearizedBasis<mpart::ProbabilistHermite>, Exp, ClenshawCurtisQuadrature, Kokkos::HostSpace)
REGISTER_MONO_COMP(LinearizedBasis<mpart::ProbabilistHermite>, SoftPlus, ClenshawCurtisQuadrature, Kokkos::HostSpace)
#if defined(MPART_ENABLE_GPU)
REGISTER_MONO_COMP(LinearizedBasis<mpart::ProbabilistHermite>, Exp, ClenshawCurtisQuadrature, mpart::DeviceSpace)
REGISTER_MONO_COMP(LinearizedBasis<mpart::ProbabilistHermite>, Softplus, ClenshawCurtisQuadrature, mpart::DeviceSpace)
#endif 
CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory14)
#endif 