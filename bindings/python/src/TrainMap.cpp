#include "CommonPybindUtilities.h"
#include "MParT/TrainMap.h"
#include "MParT/MapObjective.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

#if defined(MPART_HAS_CEREAL)
#include "MParT/Utilities/Serialization.h"
#include <fstream>
#endif // MPART_HAS_CEREAL

namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::TrainOptionsWrapper(py::module &m)
{
    // TrainOptions
    py::class_<TrainOptions, std::shared_ptr<TrainOptions>>(m, "TrainOptions")
    .def(py::init<>())
    .def("__str__", &TrainOptions::String)
    .def("__repr__", [](TrainOptions opts){return "<TrainOptions with fields\n" + opts.String() + ">";})
    .def_readwrite("opt_alg", &TrainOptions::opt_alg)
    .def_readwrite("opt_stopval", &TrainOptions::opt_stopval)
    .def_readwrite("opt_ftol_rel", &TrainOptions::opt_ftol_rel)
    .def_readwrite("opt_ftol_abs", &TrainOptions::opt_ftol_abs)
    .def_readwrite("opt_xtol_rel", &TrainOptions::opt_xtol_rel)
    .def_readwrite("opt_xtol_abs", &TrainOptions::opt_xtol_abs)
    .def_readwrite("opt_maxeval", &TrainOptions::opt_maxeval)
    .def_readwrite("opt_maxtime", &TrainOptions::opt_maxtime)
    .def_readwrite("verbose", &TrainOptions::verbose)
    ;

    
}

template<typename MemorySpace>
void mpart::binding::TrainMapWrapper(py::module &m) {

    std::string tName = "TrainMap";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    m.def(tName.c_str(), &TrainMap<Kokkos::HostSpace>)
    ;
}

template void mpart::binding::TrainMapWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::TrainMapWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU