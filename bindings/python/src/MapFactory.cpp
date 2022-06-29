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
    // CreateComponent
    m.def("CreateComponent", [] (FixedMultiIndexSet<Kokkos::HostSpace> const& mset, 
                                 MapOptions options)
    {
        return KokkosCustomPointer(MapFactory::CreateComponent(mset,options));
        //return MapFactory::CreateComponent(mset,options);
    });

    // CreateTriangular
    m.def("CreateTriangular", [] (unsigned int inputDim, 
                                  unsigned int outputDim,
                                  unsigned int totalOrder, 
                                  MapOptions options)
    {
        return KokkosCustomPointer(MapFactory::CreateTriangular(inputDim, outputDim, totalOrder, options));
    });

}