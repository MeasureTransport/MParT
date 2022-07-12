#include "CommonPybindUtilities.h"
#include "MParT/ParameterizedFunctionBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;


void mpart::binding::ParameterizedFunctionBaseWrapper(py::module &m)
{
    
    // ParameterizedFunctionBase
     py::class_<ParameterizedFunctionBase<Kokkos::HostSpace>, std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>>>(m, "ParameterizedFunctionBase")
        .def("CoeffMap", &ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffMap)
        .def("SetCoeffs", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&ParameterizedFunctionBase<Kokkos::HostSpace>::SetCoeffs))
        .def("Evaluate", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ParameterizedFunctionBase<Kokkos::HostSpace>::Evaluate))
        .def("CoeffGrad", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffGrad))
        .def_readonly("numCoeffs", &ParameterizedFunctionBase<Kokkos::HostSpace>::numCoeffs)
        .def_readonly("inputDim", &ParameterizedFunctionBase<Kokkos::HostSpace>::inputDim)
        .def_readonly("outputDim", &ParameterizedFunctionBase<Kokkos::HostSpace>::outputDim)
        ;

}