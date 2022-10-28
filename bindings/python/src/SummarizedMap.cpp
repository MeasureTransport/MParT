#include "CommonPybindUtilities.h"
#include "MParT/SummarizedMap.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/ParameterizedFunctionBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::SummarizedMapWrapper(py::module &m)
{
    std::string tName = "SummarizedMap";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    // SummarizedMap
    py::class_<SummarizedMap<MemorySpace>, ConditionalMapBase<MemorySpace>, std::shared_ptr<SummarizedMap<MemorySpace>>>(m, tName.c_str())
        .def(py::init<std::shared_ptr<ParameterizedFunctionBase<MemorySpace>>, std::shared_ptr<ConditionalMapBase<MemorySpace>>>())
        .def("print_map_ptr", &SummarizedMap<MemorySpace>::print_map_ptr)
        ;

}

template void mpart::binding::SummarizedMapWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::SummarizedMapWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU