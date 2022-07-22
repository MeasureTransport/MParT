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

#ifdef MPART_ENABLE_GPU
typedef double::false_type false_type;
template<typename MemorySpace>
void mpart::binding::ConditionalMapBaseDeviceWrapper(py::module &m)
{
    typedef double::false_type false_type;
    // ConditionalMapBaseDevice
     py::class_<ConditionalMapBase<MemorySpace>, ParameterizedFunctionBase<MemorySpace>, std::shared_ptr<ConditionalMapBase<MemorySpace>>>(m, "ConditionalMapBaseDevice")

        .def("LogDeterminant", [](ConditionalMapBase<MemorySpace> &map, DeviceMatrix<const double> &pts){ return DeviceMatrix(map.LogDeterminant(pts.data)); })
        .def("Inverse", [](ConditionalMapBase<MemorySpace> &map, DeviceMatrix<const double> const &x1, DeviceMatrix<const double> const &r){ return DeviceMatrix(map.Inverse(x1.data, r.data)); }
        .def("LogDeterminantCoeffGrad", [](ConditionalMapBase<MemorySpace> &map, DeviceMatrix<const double> const pts){ return DeviceMatrix(map.LogDeterminantCoeffGrad(pts)); }
        .def("GetBaseFunction", &ConditionalMapBase<MemorySpace>::GetBaseFunction)
        ;
}
#endif