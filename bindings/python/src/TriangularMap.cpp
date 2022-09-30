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
    py::class_<TriangularMap<MemorySpace>, ConditionalMapBase<MemorySpace>, std::shared_ptr<TriangularMap<MemorySpace>>>(m, tName.c_str())
        .def(py::init<std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>>>())
        .def("GetComponent", &TriangularMap<MemorySpace>::GetComponent)
        .def("Slice", &TriangularMap<MemorySpace>::Slice)
        .def("__getitem__", &TriangularMap<MemorySpace>::GetComponent)
        .def("__getitem__", [](const TriangularMap<MemorySpace> &m, const py::slice &s) {
            size_t start = 0, stop = 0, step = 0, slicelength = 0;
            s.compute(m.outputDim, &start, &stop, &step, &slicelength);
            if(step != 1) throw std::runtime_error("TriangularMap slice stride must be 1");
            return m.Slice(start, stop);
        })
        ;

}

template void mpart::binding::TriangularMapWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::TriangularMapWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU