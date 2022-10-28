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
    DebugMapWrapper<Kokkos::HostSpace>(m);
    MapFactoryWrapper<Kokkos::HostSpace>(m);



#if defined(MPART_ENABLE_GPU)
    ParameterizedFunctionBaseWrapper<mpart::DeviceSpace>(m);
    ConditionalMapBaseWrapper<mpart::DeviceSpace>(m);
    TriangularMapWrapper<mpart::DeviceSpace>(m);
    MapFactoryWrapper<mpart::DeviceSpace>(m);
    AffineMapWrapperDevice(m);
    AffineFunctionWrapperDevice(m);
#endif
}