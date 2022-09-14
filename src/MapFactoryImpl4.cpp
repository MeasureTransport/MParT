#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

template<typename MemorySpace, typename PosFuncType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_Prob_ACC(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    ProbabilistHermite basis1d(opts.basisNorm);

    unsigned int level = std::log2(opts.quadPts-2);
    AdaptiveClenshawCurtis<MemorySpace> quad(level, opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);

    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
    return output;
}

static auto reg_host_prob_acc_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::ProbabilistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis), CreateComponentImpl_Prob_ACC<Kokkos::HostSpace, Exp>));
static auto reg_host_prob_acc_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::ProbabilistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis), CreateComponentImpl_Prob_ACC<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_prob_acc_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::ProbabilistHermite, false, PosFuncTypes::Exp, QuadTypes::AdaptiveClenshawCurtis), CreateComponentImpl_Prob_ACC<mpart::DeviceSpace, Exp>));
    static auto reg_device_prob_acc_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::ProbabilistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::AdaptiveClenshawCurtis), CreateComponentImpl_Prob_ACC<mpart::DeviceSpace, SoftPlus>));
#endif