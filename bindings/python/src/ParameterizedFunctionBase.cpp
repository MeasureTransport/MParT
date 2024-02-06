#include "CommonPybindUtilities.h"
#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/Utilities/ArrayConversions.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

#if defined(MPART_HAS_CEREAL)
#include <fstream>
#endif

namespace py = pybind11;
using namespace mpart::binding;

template<>
void mpart::binding::ParameterizedFunctionBaseWrapper<Kokkos::HostSpace>(py::module &m)
{
    // ParameterizedFunctionBase
    py::class_<ParameterizedFunctionBase<Kokkos::HostSpace>, std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>>>(m, "ParameterizedFunctionBase")
        .def("CoeffMap", &ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffMap)
        .def("SetCoeffs", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&ParameterizedFunctionBase<Kokkos::HostSpace>::SetCoeffs))
        .def("WrapCoeffs", [](std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> obj, std::tuple<long,int,int> coeffs){
            obj->WrapCoeffs(ToKokkos<double,Kokkos::HostSpace>(coeffs));
        })
        .def("Evaluate", static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<Kokkos::HostSpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<Kokkos::HostSpace>::Evaluate))
        .def("EvaluateImpl", [](std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->EvaluateImpl(ToKokkos<double,Kokkos::HostSpace>(input),ToKokkos<double,Kokkos::HostSpace>(output));
        })
        .def("Gradient", static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<Kokkos::HostSpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<Kokkos::HostSpace>::Gradient))
        .def("GradientImpl", [](std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> sens, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->GradientImpl(ToKokkos<double,Kokkos::HostSpace>(input),ToKokkos<double,Kokkos::HostSpace>(sens), ToKokkos<double,Kokkos::HostSpace>(output));
        })
        .def("CoeffGrad",static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<Kokkos::HostSpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<Kokkos::HostSpace>::CoeffGrad))
        .def("CoeffGradImpl",[](std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> sens, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->CoeffGradImpl(ToKokkos<double,Kokkos::HostSpace>(input),ToKokkos<double,Kokkos::HostSpace>(sens), ToKokkos<double,Kokkos::HostSpace>(output));
        })
        .def("torch", [](std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> obj, bool store_coeffs){
            auto mpart = py::module::import("mpart");
            if(!mpart.attr("mpart_has_torch").cast<bool>()){
                throw std::runtime_error("MParT could not import pytorch.");
            }
            return mpart.attr("TorchParameterizedFunctionBase")(obj, store_coeffs);
        }, py::arg("store_coeffs")=true)
        .def("DiagonalCoeffIndices", &ParameterizedFunctionBase<Kokkos::HostSpace>::DiagonalCoeffIndices)
        .def_readonly("numCoeffs", &ParameterizedFunctionBase<Kokkos::HostSpace>::numCoeffs)
        .def_readonly("inputDim", &ParameterizedFunctionBase<Kokkos::HostSpace>::inputDim)
        .def_readonly("outputDim", &ParameterizedFunctionBase<Kokkos::HostSpace>::outputDim)
#if defined(MPART_HAS_CEREAL)
        .def("Serialize", [](ParameterizedFunctionBase<Kokkos::HostSpace> const &obj, std::string const &filename){
            std::ofstream os(filename);
            cereal::BinaryOutputArchive archive(os);
            archive(obj.inputDim, obj.outputDim, obj.numCoeffs);
            archive(obj.Coeffs());
        })
        .def("ToBytes", [](std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> const &ptr) {
            std::stringstream ss;
            ptr->Save(ss);
            return py::bytes(ss.str());
        })
        .def_static("FromBytes", [](std::string input) {
            std::stringstream ss;
            ss.str(input);

            auto ptr = ParameterizedFunctionBase<Kokkos::HostSpace>::Load(ss);
            return ptr;  
        })
        .def(py::pickle(
            [](std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> const& ptr) { // __getstate__
                std::stringstream ss;
                ptr->Save(ss);
                return py::bytes(ss.str());
            },
            [](py::bytes input) {
                
                std::stringstream ss;
                ss.str(input);

                auto ptr = ParameterizedFunctionBase<Kokkos::HostSpace>::Load(ss);
                return ptr;
            }
        ))
        #endif
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