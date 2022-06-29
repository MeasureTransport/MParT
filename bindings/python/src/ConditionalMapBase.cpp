#include "CommonPybindUtilities.h"
#include "MParT/ConditionalMapBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;


// class PyConditionalMapBase : public ConditionalMapBase
// {
// public:

//     using ConditionalMapBase::ConditionalMapBase;

// };

void mpart::binding::ConditionalMapBaseWrapper(py::module &m)
{

    // ConditionalMapBase
     py::class_<ConditionalMapBase<Kokkos::HostSpace>, KokkosCustomPointer<ConditionalMapBase<Kokkos::HostSpace>>>(m, "ConditionalMapBase")

        .def("CoeffMap", &ConditionalMapBase<Kokkos::HostSpace>::CoeffMap)
        .def("SetCoeffs", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&ConditionalMapBase<Kokkos::HostSpace>::SetCoeffs))
        .def("Evaluate", py::overload_cast<Eigen::RowMatrixXd const&>(&ConditionalMapBase<Kokkos::HostSpace>::Evaluate))
        .def("LogDeterminant", py::overload_cast<Eigen::RowMatrixXd const&>(&ConditionalMapBase<Kokkos::HostSpace>::LogDeterminant))
        .def("Inverse", py::overload_cast<Eigen::RowMatrixXd const&, Eigen::RowMatrixXd const&>(&ConditionalMapBase<Kokkos::HostSpace>::Inverse))
        .def_readonly("numCoeffs", &ConditionalMapBase<Kokkos::HostSpace>::numCoeffs)
        ;

}