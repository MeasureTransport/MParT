#include "CommonPybindUtilities.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "MParT/Initialization.h"
#include <Kokkos_Core.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

// Define a wrapper around Kokkos::Initialize that accepts a python dictionary instead of argc and argv.
void mpart::binding::Initialize(py::dict opts) {

    Kokkos::InitArguments args;

    pybind11::object keys = pybind11::list(opts.attr("keys")());
    std::vector<std::string> keysCpp = keys.cast<std::vector<std::string>>();

    for(auto& key : keysCpp){
        
        if((key=="num_threads")||(key=="kokkos-threads")){
            args.num_threads = py::cast<int>(opts.attr("get")(key));
        }else if(key=="num_numa"){
            args.num_numa = py::cast<int>(opts.attr("get")(key));
        }else if(key=="device_id"){
            args.device_id = py::cast<int>(opts.attr("get")(key));
        }else if(key=="ndevices"){
            args.ndevices = py::cast<int>(opts.attr("get")(key));
        }else if(key=="skip_device"){
            args.skip_device = py::cast<int>(opts.attr("get")(key));
        }else if(key=="disable_warnings"){
            args.disable_warnings = py::cast<bool>(opts.attr("get")(key));
        }else{
            std::cout << "WARNING: Kokkos initialization argument \"" << key << "\" was not used." << std::endl;
        }
    }

    mpart::Initialize(args);
};

void mpart::binding::CommonUtilitiesWrapper(py::module &m)
{   
    m.def("Initialize", py::overload_cast<py::dict>( &mpart::binding::Initialize ));
    m.def("Concurrency", &Kokkos::DefaultExecutionSpace::concurrency);
}
