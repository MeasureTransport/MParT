#include "CommonPybindUtilities.h"
#include "MParT/ConditionalMapBase.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::ConditionalMapBaseWrapper(py::module &m)
{
    std::string tName = "ConditionalMapBase";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    // ConditionalMapBase
    py::class_<ConditionalMapBase<MemorySpace>, ParameterizedFunctionBase<MemorySpace>, std::shared_ptr<ConditionalMapBase<MemorySpace>>>(m, tName.c_str())

        .def("LogDeterminant", static_cast<Eigen::VectorXd (ConditionalMapBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ConditionalMapBase<MemorySpace>::LogDeterminant))
        .def("LogDeterminantImpl", [](std::shared_ptr<ConditionalMapBase<MemorySpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,int,int> output){
            obj->LogDeterminantImpl(ToKokkos<double,MemorySpace>(input),ToKokkos<double,MemorySpace>(output));
        })
        .def("Inverse", static_cast<Eigen::RowMatrixXd (ConditionalMapBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ConditionalMapBase<MemorySpace>::Inverse))
        .def("LogDeterminantCoeffGrad", static_cast<Eigen::RowMatrixXd (ConditionalMapBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ConditionalMapBase<MemorySpace>::LogDeterminantCoeffGrad))
        .def("LogDeterminantCoeffGradImpl", [](std::shared_ptr<ConditionalMapBase<MemorySpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->LogDeterminantCoeffGradImpl(ToKokkos<double,MemorySpace>(input),ToKokkos<double,MemorySpace>(output));
        })
        .def("LogDeterminantInputGrad", static_cast<Eigen::RowMatrixXd (ConditionalMapBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ConditionalMapBase<MemorySpace>::LogDeterminantInputGrad))
        .def("LogDeterminantInputGradImpl", [](std::shared_ptr<ConditionalMapBase<MemorySpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->LogDeterminantInputGradImpl(ToKokkos<double,MemorySpace>(input),ToKokkos<double,MemorySpace>(output));
        })
        .def("torch", [](std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> obj, bool return_logdet){
            return py::module::import("mpart").attr("TorchConditionalMapBase")(obj, return_logdet);
        }, py::arg("return_logdet") = false)
        .def("GetBaseFunction", &ConditionalMapBase<MemorySpace>::GetBaseFunction)
#if defined(MPART_HAS_CEREAL)
        .def(py::pickle(
            [](std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> const& ptr) { // __getstate__
                std::stringstream ss;
                ptr->Save(ss);
                return py::bytes(ss.str());
            },
            [](py::bytes input) {
                
                std::stringstream ss;
                ss.str(input);

                auto ptr = std::dynamic_pointer_cast<ConditionalMapBase<Kokkos::HostSpace>>(ParameterizedFunctionBase<Kokkos::HostSpace>::Load(ss));
                return ptr;
            }
        ))
#endif
        ;

}

template void mpart::binding::ConditionalMapBaseWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::ConditionalMapBaseWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU