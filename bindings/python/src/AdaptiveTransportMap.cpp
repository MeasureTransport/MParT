#include "CommonPybindUtilities.h"
#include "MParT/MapOptions.h"
#include "MParT/MapObjective.h"
#include "MParT/TrainMap.h"
#include "MParT/AdaptiveTransportMap.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::ATMOptionsWrapper(py::module &m)
{
    // ATMOptions
    py::class_<ATMOptions, MapOptions, TrainOptions, std::shared_ptr<ATMOptions>>(m, "ATMOptions")
    .def(py::init<>())
    .def("__str__", &ATMOptions::String)
    .def("__repr__", [](ATMOptions opts){return "<ATMOptions with fields\n" + opts.String() + ">";})
    .def_readwrite("maxPatience", &ATMOptions::maxPatience)
    .def_readwrite("maxSize", &ATMOptions::maxSize)
    .def_readwrite("maxDegrees", &ATMOptions::maxDegrees)
    ;

}

template<typename MemorySpace>
void mpart::binding::AdaptiveTransportMapWrapper(py::module &m) {

    std::string tName = "AdaptiveTransportMap";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    m.def(tName.c_str(), &AdaptiveTransportMap<MemorySpace>)
    ;
}

template void mpart::binding::AdaptiveTransportMapWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::AdaptiveTransportMapWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU