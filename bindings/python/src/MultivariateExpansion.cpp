#include "CommonPybindUtilities.h"
#include "MParT/MultivariateExpansion.h"
#include "MParT/OrthogonalPolynomial.h"
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
    py::class_<MultivariateExpansion<ProbabilistHermite,Kokkos::HostSpace>>(m, "MultivariateExpansion")
        // MultivariateExpansion
        .def(py::init<MultiIndexSet const&>())
        .def("NumCoeffs", &MultivariateExpansion<ProbabilistHermite,Kokkos::HostSpace>::NumCoeffs)
        //.def("Evaluate", &MultivariateExpansion<ProbabilistHermite,Kokkos::HostSpace>::Evaluate)
        ;

}