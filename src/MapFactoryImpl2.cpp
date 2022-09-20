#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;


template<typename MemorySpace, typename PosFuncType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_Phys_CC(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    PhysicistHermite basis1d(opts.basisNorm);
    ClenshawCurtisQuadrature<MemorySpace> quad(opts.quadPts, 1);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);

    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
    return output;
}

static auto reg_host_phys_cc_exp = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis), CreateComponentImpl_Phys_CC<Kokkos::HostSpace, Exp>));
static auto reg_host_phys_cc_splus = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis), CreateComponentImpl_Phys_CC<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_phys_cc_exp = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis), CreateComponentImpl_Phys_CC<mpart::DeviceSpace, Exp>));
    static auto reg_device_phys_cc_splus = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::PhysicistHermite, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis), CreateComponentImpl_Phys_CC<mpart::DeviceSpace, SoftPlus>));
#endif