#include "CommonPybindUtilities.h"
#include <MParT/Utilities/GPUtils.h>

using namespace mpart::binding;


PYBIND11_MODULE(pympart, m) {

    CommonUtilitiesWrapper(m);
    MultiIndexWrapper(m);
    MapOptionsWrapper(m);

    ParameterizedFunctionBaseWrapper<Kokkos::HostSpace>(m);
    ConditionalMapBaseWrapper<Kokkos::HostSpace>(m);
    AffineMapWrapperHost(m);
    AffineFunctionWrapperHost(m);
    TriangularMapWrapper<Kokkos::HostSpace>(m);
    ComposedMapWrapper<Kokkos::HostSpace>(m);
    SummarizedMapWrapper<Kokkos::HostSpace>(m);
    IdentityMapWrapper<Kokkos::HostSpace>(m);
    // DebugMapWrapper<Kokkos::HostSpace>(m);

#if defined(MPART_HAS_NLOPT)
    MapObjectiveWrapper<Kokkos::HostSpace>(m);
    TrainOptionsWrapper(m);
    TrainMapWrapper<Kokkos::HostSpace>(m);
    MapFactoryWrapper<Kokkos::HostSpace>(m);
#endif // MPART_HAS_NLOPT

#if defined(MPART_HAS_CEREAL)
    DeserializeWrapper<Kokkos::HostSpace>(m);
#endif // MPART_HAS_CEREAL

#if defined(MPART_ENABLE_GPU)
    ParameterizedFunctionBaseWrapper<mpart::DeviceSpace>(m);
    ConditionalMapBaseWrapper<mpart::DeviceSpace>(m);
    TriangularMapWrapper<mpart::DeviceSpace>(m);
    MapFactoryWrapper<mpart::DeviceSpace>(m);
    AffineMapWrapperDevice(m);
    AffineFunctionWrapperDevice(m);
    IdentityMapWrapper<Kokkos::DeviceSpace>(m);
    SerializeWrapper<mpart::DeviceSpace>(m);
    DeserializeWrapper<mpart::DeviceSpace>(m);
#endif
}