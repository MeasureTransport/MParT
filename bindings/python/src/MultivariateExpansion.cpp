#include "CommonPybindUtilities.h"
#include "MParT/MultivariateExpansion.h"
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::MultivariateExpansionWrapper(py::module &m)
{
    py::class_<MultivariateExpansion<BasisEvaluatorType, Kokkos::HostSpace>, KokkosCustomPointer<MultivariateExpansion>>(m, "MultivariateExpansion")
        // MultivariateExpansion
        .def(py::init<MultiIndexSet const&>())
        .def("Evaluate", &MultivariateExpansion::Evaluate)
        ;

}