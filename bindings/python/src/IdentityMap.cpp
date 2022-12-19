#include "CommonPybindUtilities.h"
#include "MParT/IdentityMap.h"
#include "MParT/ConditionalMapBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::IdentityMapWrapper(py::module &m)
{
    std::string tName = "IdentityMap";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    // IdentityMap
    py::class_<IdentityMap<MemorySpace>, ConditionalMapBase<MemorySpace>, std::shared_ptr<IdentityMap<MemorySpace>>>(m, tName.c_str())
        .def(py::init<unsigned int, unsigned int>())
        ;

}

template void mpart::binding::IdentityMapWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::IdentityMapWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU