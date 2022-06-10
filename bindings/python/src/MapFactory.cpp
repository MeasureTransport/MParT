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
    // CteateComponent
    m.def("CreateComponent", &MapFactory::CreateComponent);
    // m.def("CreateComponent", [] (FixedMultiIndexSet<Kokkos::HostSpace> const& mset, 
    //                              MapOptions options = MapOptions())
    // {
    //     return CustomPtrToSharedPtr(MapFactory::CreateComponent(mset,options));
    // }
    
    
    
    ;

}