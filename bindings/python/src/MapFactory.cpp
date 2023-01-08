#include "CommonPybindUtilities.h"
#include "MParT/MapFactory.h"
#include "MParT/ConditionalMapBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::MapFactoryWrapper(py::module &m)
{
    bool isDevice = !std::is_same<MemorySpace,Kokkos::HostSpace>::value;
    // CreateComponent
    m.def(isDevice? "dCreateComponent" : "CreateComponent", &MapFactory::CreateComponent<MemorySpace>);

    // CreateTriangular
    m.def(isDevice? "dCreateTriangular" : "CreateTriangular", &MapFactory::CreateTriangular<MemorySpace>);

    // CreateSingleEntryMap
    m.def(isDevice? "dCreateSingleEntryMap" : "CreateSingleEntryMap", &MapFactory::CreateSingleEntryMap<MemorySpace>);
}
template void mpart::binding::MapFactoryWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::MapFactoryWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU