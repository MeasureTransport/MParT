#include "CommonPybindUtilities.h"
#include "MParT/ParameterizedFunctionBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::ParameterizedFunctionBaseWrapper(py::module &m)
{
    std::string tName = "ParameterizedFunctionBase";
    constexpr if(std::is_same<MemorySpace,DeviceSpace>::value) tName = "ParameterizedFunctionBaseDevice";
    // ParameterizedFunctionBase
    py::class_<ParameterizedFunctionBase<MemorySpace>, std::shared_ptr<ParameterizedFunctionBase<MemorySpace>>>(m, tName)
        .def("CoeffMap", &ParameterizedFunctionBase<MemorySpace>::CoeffMap)
        .def("SetCoeffs", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&ParameterizedFunctionBase<MemorySpace>::SetCoeffs))
        .def("Evaluate", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ParameterizedFunctionBase<MemorySpace>::Evaluate))
        .def("CoeffGrad", py::overload_cast<Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&>(&ParameterizedFunctionBase<MemorySpace>::CoeffGrad))
        .def_readonly("numCoeffs", &ParameterizedFunctionBase<MemorySpace>::numCoeffs)
        .def_readonly("inputDim", &ParameterizedFunctionBase<MemorySpace>::inputDim)
        .def_readonly("outputDim", &ParameterizedFunctionBase<MemorySpace>::outputDim)
        ;

}

template void mpart::binding::ParameterizedFunctionBaseWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::ParameterizedFunctionBaseWrapper<DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU