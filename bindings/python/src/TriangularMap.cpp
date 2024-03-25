#include "CommonPybindUtilities.h"
#include "MParT/TriangularMap.h"
#include "MParT/ConditionalMapBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::TriangularMapWrapper(py::module &m)
{
    std::string tName = "TriangularMap";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    // TriangularMap
    py::class_<TriangularMap<MemorySpace>, ConditionalMapBase<MemorySpace>, ParameterizedFunctionBase<MemorySpace>, std::shared_ptr<TriangularMap<MemorySpace>>>(m, tName.c_str())
        .def(py::init<std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>>, bool>(), py::arg("comps"), py::arg("moveCoeffs") = false)
        .def("GetComponent", &TriangularMap<MemorySpace>::GetComponent)
#if defined(MPART_HAS_CEREAL)
        .def(py::pickle(
            [](std::shared_ptr<TriangularMap<MemorySpace>> const& ptr) { // __getstate__
                std::stringstream ss;
                ptr->Save(ss);
                return py::bytes(ss.str());
            },
            [](py::bytes input) {
                
                std::stringstream ss;
                ss.str(input);

                auto ptr = std::dynamic_pointer_cast<TriangularMap<MemorySpace>>(ParameterizedFunctionBase<MemorySpace>::Load(ss));
                return ptr;
            }
        ))
#endif
    ;

}

template void mpart::binding::TriangularMapWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::TriangularMapWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU