#include "CommonPybindUtilities.h"

using namespace mpart::binding;


PYBIND11_MODULE(pympart, m) {

    CommonUtilitiesWrapper(m);
    MultiIndexWrapper(m);
    MapOptionsWrapper(m);

    ParameterizedFunctionBaseWrapper<Kokkos::HostSpace>(m);
    ConditionalMapBaseWrapper<Kokkos::HostSpace>(m);
    TriangularMapWrapper<Kokkos::HostSpace>(m);
    MapFactoryWrapper<Kokkos::HostSpace>(m);

#ifdef MPART_ENABLE_GPU
    ParameterizedFunctionBaseWrapper<DeviceSpace>(m);
    ConditionalMapBaseWrapper<DeviceSpace>(m);
    TriangularMapWrapper<DeviceSpace>(m);
    MapFactoryWrapper<DeviceSpace>(m);
#endif
}