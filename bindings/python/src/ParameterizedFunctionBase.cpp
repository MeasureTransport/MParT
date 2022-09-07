#include "CommonPybindUtilities.h"
#include "MParT/ParameterizedFunctionBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<>
void mpart::binding::ParameterizedFunctionBaseWrapper<Kokkos::HostSpace>(py::module &m)
{
    // ParameterizedFunctionBase
    py::class_<ParameterizedFunctionBase<Kokkos::HostSpace>, std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>>>(m, "ParameterizedFunctionBase")
        .def("CoeffMap", &ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffMap)
        .def("SetCoeffs", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&ParameterizedFunctionBase<Kokkos::HostSpace>::SetCoeffs))
        .def("Evaluate", static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<Kokkos::HostSpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<Kokkos::HostSpace>::Evaluate))
        .def("Gradient", static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<Kokkos::HostSpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<Kokkos::HostSpace>::Gradient))
        .def("CoeffGrad",static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<Kokkos::HostSpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffGrad))
        .def_readonly("numCoeffs", &ParameterizedFunctionBase<Kokkos::HostSpace>::numCoeffs)
        .def_readonly("inputDim", &ParameterizedFunctionBase<Kokkos::HostSpace>::inputDim)
        .def_readonly("outputDim", &ParameterizedFunctionBase<Kokkos::HostSpace>::outputDim)
        ;
}

#if defined(MPART_ENABLE_GPU)
template<>
void mpart::binding::ParameterizedFunctionBaseWrapper<mpart::DeviceSpace>(py::module &m)
{

    // ParameterizedFunctionBase
    py::class_<ParameterizedFunctionBase<mpart::DeviceSpace>, std::shared_ptr<ParameterizedFunctionBase<mpart::DeviceSpace>>>(m, "dParameterizedFunctionBase")
        .def("CoeffMap", [](const ParameterizedFunctionBase<mpart::DeviceSpace> &f) {
            Kokkos::View<const double*, Kokkos::HostSpace> host_coeffs = ToHost<mpart::DeviceSpace, const double*>( f.Coeffs() );
            return Eigen::VectorXd(Eigen::Map<const Eigen::VectorXd>(host_coeffs.data(), host_coeffs.size()));
        })
        .def("SetCoeffs", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&ParameterizedFunctionBase<mpart::DeviceSpace>::SetCoeffs))
        .def("Evaluate", static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<mpart::DeviceSpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<mpart::DeviceSpace>::Evaluate))
        .def("Gradient", static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<mpart::DeviceSpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<mpart::DeviceSpace>::Gradient))
        .def("CoeffGrad",static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<mpart::DeviceSpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<mpart::DeviceSpace>::CoeffGrad))
        .def_readonly("numCoeffs", &ParameterizedFunctionBase<mpart::DeviceSpace>::numCoeffs)
        .def_readonly("inputDim", &ParameterizedFunctionBase<mpart::DeviceSpace>::inputDim)
        .def_readonly("outputDim", &ParameterizedFunctionBase<mpart::DeviceSpace>::outputDim)
        ;
}

#endif // MPART_ENABLE_GPU