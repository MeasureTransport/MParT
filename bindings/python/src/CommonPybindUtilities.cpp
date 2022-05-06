#include "CommonPybindUtilities.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

// Define a wrapper around Kokkos::Initialize that accepts a python dictionary instead of argc and argv.
KokkosRuntime mpart::binding::KokkosInit(py::dict opts) {

    std::vector<std::string> args;

    pybind11::object keys = pybind11::list(opts.attr("keys")());
    std::vector<std::string> keysCpp = keys.cast<std::vector<std::string>>();

    for(auto& key : keysCpp){
        std::string val = "--" + key + "=";
        val += (std::string) pybind11::str(opts.attr("get")(key));
        args.push_back(val);
    }
    return KokkosInit(args);
};


void mpart::binding::CommonUtilitiesWrapper(py::module &m)
{
    py::class_<KokkosRuntime, KokkosCustomPointer<KokkosRuntime>>(m, "KokkosRuntime")
        .def(py::init<>())
        .def("ElapsedTime", &KokkosRuntime::ElapsedTime);
    
    m.def("KokkosInit", py::overload_cast<py::dict>( &KokkosInit ));
    m.def("KokkosInit", py::overload_cast<std::vector<std::string>>( &KokkosInit ));
    m.def("KokkosFinalize", &Kokkos::finalize);
}
