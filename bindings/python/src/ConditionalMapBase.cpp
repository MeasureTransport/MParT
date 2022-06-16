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
     py::class_<ConditionalMapBase, KokkosCustomPointer<ConditionalMapBase>>(m, "ConditionalMapBase")
    //py::class_<ConditionalMapBase, std::shared_ptr<ConditionalMapBase>>(m, "ConditionalMapBase")
        .def("CoeffMap", &ConditionalMapBase::CoeffMap)
        .def("SetCoeffs", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&ConditionalMapBase::SetCoeffs))
        //.def("savedCoeffs", &ConditionalMapBase::savedCoeffs)
        .def("Evaluate", py::overload_cast<Eigen::RowMatrixXd const&>(&ConditionalMapBase::Evaluate))
        .def("LogDeterminant", py::overload_cast<Eigen::RowMatrixXd const&>(&ConditionalMapBase::LogDeterminant))
        .def("Inverse", py::overload_cast<Eigen::RowMatrixXd const&, Eigen::RowMatrixXd const&>(&ConditionalMapBase::Inverse))
        ;

}