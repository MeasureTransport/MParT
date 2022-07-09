#include "CommonPybindUtilities.h"
#include "MParT/TriangularMap.h"
#include "MParT/ConditionalMapBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::TriangularMapWrapper(py::module &m)
{

    // TriangularMap
     py::class_<TriangularMap<Kokkos::HostSpace>, ConditionalMapBase<Kokkos::HostSpace>, std::shared_ptr<TriangularMap<Kokkos::HostSpace>>>(m, "TriangularMap")
        .def(py::init<std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>>>())
        .def("InverseInplace", &TriangularMap<Kokkos::HostSpace>::InverseInplace)
        .def("GetComponent", &TriangularMap<Kokkos::HostSpace>::GetComponent)
        ;

}