#include "CommonPybindUtilities.h"
#include "MParT/DebugMap.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/ParameterizedFunctionBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::DebugMapWrapper(py::module &m)
{
    std::string tName = "DebugMap";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    // DebugMap
    py::class_<DebugMap<MemorySpace>, ConditionalMapBase<MemorySpace>, std::shared_ptr<DebugMap<MemorySpace>>>(m, tName.c_str())
        .def(py::init<std::shared_ptr<ConditionalMapBase<MemorySpace>>>())
        .def("print_ptr", &DebugMap<MemorySpace>::print_ptr)
        ;

}

template void mpart::binding::DebugMapWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::DebugMapWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU