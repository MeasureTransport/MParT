#include "CommonPybindUtilities.h"
#include <MParT/Utilities/GPUtils.h>

using namespace mpart::binding;


PYBIND11_MODULE(pympart, m) {

    CommonUtilitiesWrapper(m);
    MultiIndexWrapper(m);
    MapOptionsWrapper(m);

    FixedMultiIndexSetWrapper<Kokkos::HostSpace>(m);
    ParameterizedFunctionBaseWrapper<Kokkos::HostSpace>(m);
    ConditionalMapBaseWrapper<Kokkos::HostSpace>(m);
    TriangularMapWrapper<Kokkos::HostSpace>(m);
    MapFactoryWrapper<Kokkos::HostSpace>(m);

#if defined(MPART_ENABLE_GPU)
    FixedMultiIndexSetWrapper<mpart::DeviceSpace>(m);
    ParameterizedFunctionBaseWrapper<mpart::DeviceSpace>(m);
    ConditionalMapBaseWrapper<mpart::DeviceSpace>(m);
    TriangularMapWrapper<mpart::DeviceSpace>(m);
    MapFactoryWrapper<mpart::DeviceSpace>(m);
#endif
}