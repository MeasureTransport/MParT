#include "CommonPybindUtilities.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::python;


KokkosGuard::KokkosGuard()  
{ 

}

KokkosGuard::~KokkosGuard() 
{ 
    Kokkos::finalize(); 
}

std::shared_ptr<KokkosGuard> mpart::python::GetKokkosGuard()
{
  static std::weak_ptr<KokkosGuard> kokkos_guard_;
  auto shared = kokkos_guard_.lock();
  if (!shared)
  {
    shared = std::make_shared<KokkosGuard>();
    kokkos_guard_ = shared;
  }
  return shared;
}


KokkosRuntime::KokkosRuntime() : start(std::chrono::high_resolution_clock::now()) {};

double KokkosRuntime::ElapsedTime() const{
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count() / 1e6;
};

// Define a wrapper around Kokkos::Initialize that accepts a vector of strings instead of argc and argv.
KokkosRuntime mpart::python::KokkosInit(std::vector<std::string> args) {

    std::vector<char *> cstrs;
    cstrs.reserve(args.size());
    for (auto &s : args) cstrs.push_back(const_cast<char *>(s.c_str()));

    int size = cstrs.size();
    Kokkos::initialize(size, cstrs.data());

    return KokkosRuntime();
};

// Define a wrapper around Kokkos::Initialize that accepts a python dictionary instead of argc and argv.
KokkosRuntime mpart::python::KokkosInit(py::dict opts) {

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


void mpart::python::CommonUtilitiesWrapper(py::module &m)
{
    py::class_<KokkosRuntime, KokkosCustomPointer<KokkosRuntime>>(m, "KokkosRuntime")
        .def(py::init<>())
        .def("ElapsedTime", &KokkosRuntime::ElapsedTime);
    
    m.def("KokkosInit", py::overload_cast<py::dict>( &KokkosInit ));
    m.def("KokkosInit", py::overload_cast<std::vector<std::string>>( &KokkosInit ));
    m.def("KokkosFinalize", &Kokkos::finalize);
}
