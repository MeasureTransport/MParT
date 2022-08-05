#include "CommonPybindUtilities.h"
#include "MParT/MapFactory.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::MapFactoryWrapper(py::module &m)
{
    // CreateComponent
    m.def("CreateComponent", &MapFactory::CreateComponent<MemorySpace>);

    // CreateTriangular
    m.def("CreateTriangular", &MapFactory::CreateTriangular<MemorySpace>);

}

template void mpart::binding::MapFactoryWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::MapFactoryWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU