#include "CommonPybindUtilities.h"
#include "MParT/ComposedMap.h"
#include "MParT/ConditionalMapBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::ComposedMapWrapper(py::module &m)
{
    std::string tName = "ComposedMap";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    // ComposedMap
    py::class_<ComposedMap<MemorySpace>, ConditionalMapBase<MemorySpace>, std::shared_ptr<ComposedMap<MemorySpace>>>(m, tName.c_str())
        .def(py::init<std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>>,int>())
        ;

}

template void mpart::binding::ComposedMapWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::ComposedMapWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU