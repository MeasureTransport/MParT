#include "CommonPybindUtilities.h"
#include "MParT/ConditionalMapBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;


void mpart::binding::ConditionalMapBaseWrapper(py::module &m)
{

    // ConditionalMapBase
    py::class_<ConditionalMapBase<Kokkos::HostSpace>, ParameterizedFunctionBase<Kokkos::HostSpace>, std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>>(m, "ConditionalMapBase")

        .def("LogDeterminant", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ConditionalMapBase<Kokkos::HostSpace>::LogDeterminant))
        .def("Inverse", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ConditionalMapBase<Kokkos::HostSpace>::Inverse))
        .def("LogDeterminantCoeffGrad", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ConditionalMapBase<Kokkos::HostSpace>::LogDeterminantCoeffGrad))
        .def("GetBaseFunction", &ConditionalMapBase<Kokkos::HostSpace>::GetBaseFunction)
        ;

}

#if defined(MPART_ENABLE_GPU)

void mpart::binding::ConditionalMapBaseDeviceWrapper(py::module &m)
{
    // ConditionalMapBaseDevice
     py::class_<ConditionalMapBase<DeviceSpace>, ParameterizedFunctionBase<DeviceSpace>, std::shared_ptr<ConditionalMapBase<DeviceSpace>>>(m, "ConditionalMapBaseDevice")


        .def("LogDeterminant", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ConditionalMapBase<DeviceSpace>::LogDeterminant))
        .def("Inverse", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ConditionalMapBase<DeviceSpace>::Inverse))
        .def("LogDeterminantCoeffGrad", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ConditionalMapBase<DeviceSpace>::LogDeterminantCoeffGrad))
        .def("GetBaseFunction", &ConditionalMapBase<DeviceSpace>::GetBaseFunction)
        ;
}
#endif