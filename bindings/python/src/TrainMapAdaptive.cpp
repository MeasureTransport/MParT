#include "CommonPybindUtilities.h"
#include "MParT/MapOptions.h"
#include "MParT/MapObjective.h"
#include "MParT/TrainMap.h"
#include "MParT/TrainMapAdaptive.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;
using namespace mpart;

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
std::shared_ptr<ConditionalMapBase<MemorySpace>> TrainMapAdaptiveSmartPointerPython(std::vector<std::shared_ptr<MultiIndexSet>> &mset0,
                                                  std::shared_ptr<MapObjective<Kokkos::HostSpace>> objective,
                                                  ATMOptions options) 
{   
    // Create a version of the mset vector with objects not pointers
    std::vector<MultiIndexSet> newmset;
    for(auto& ptr: mset0)
        newmset.push_back(*ptr);

    // Call AdaptiveTrainMap 
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map = TrainMapAdaptive<MemorySpace>(newmset, objective, options);

    // Now update the pointers
    for(int i=0; i<mset0.size(); ++i)
        *mset0[i] = newmset[i];
    
    return map;
}

template<typename MemorySpace>
void mpart::binding::TrainMapAdaptiveWrapper(py::module &m) {

    std::string tName = "TrainMapAdaptive";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    m.def(tName.c_str(), &TrainMapAdaptiveSmartPointerPython<MemorySpace>);
}

template void mpart::binding::TrainMapAdaptiveWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::TrainMapAdaptiveWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU