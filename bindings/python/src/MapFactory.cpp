#include "CommonPybindUtilities.h"
#include "MParT/MapFactory.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::MapFactoryWrapper(py::module &m)
{
    // CreateComponent
    m.def("CreateComponent", &MapFactory::CreateComponent<Kokkos::HostSpace>);

    // CreateTriangular
    m.def("CreateTriangular", &MapFactory::CreateTriangular<Kokkos::HostSpace>);

}