#include "CommonPybindUtilities.h"
#include "MParT/AffineMap.h"
#include "MParT/AffineFunction.h"
#include <pybind11/eigen.h>

#include "MParT/Utilities/GPUtils.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::AffineMapWrapperHost(py::module &m)
{
    py::class_<AffineMap<Kokkos::HostSpace>, ConditionalMapBase<Kokkos::HostSpace>, std::shared_ptr<AffineMap<Kokkos::HostSpace>>>(m, "AffineMap")
        .def(py::init( [](Eigen::Ref<Eigen::VectorXd> const& b)
        {
            return new AffineMap<Kokkos::HostSpace>(VecToKokkos<double, Kokkos::HostSpace>(b));
        }))
        .def(py::init( [](Eigen::Ref<Eigen::MatrixXd> const& A, Eigen::Ref<Eigen::VectorXd> const& b)
        {
            return new AffineMap<Kokkos::HostSpace>(MatToKokkos<double, Kokkos::HostSpace>(A), VecToKokkos<double, Kokkos::HostSpace>(b));
        }))
        .def(py::init( [](Eigen::Ref<Eigen::MatrixXd> const& A)
        {
            return new AffineMap<Kokkos::HostSpace>(MatToKokkos<double, Kokkos::HostSpace>(A));
        }));
}


void mpart::binding::AffineFunctionWrapperHost(py::module &m)
{
    py::class_<AffineFunction<Kokkos::HostSpace>, ParameterizedFunctionBase<Kokkos::HostSpace>, std::shared_ptr<AffineFunction<Kokkos::HostSpace>>>(m, "AffineFunction")
        .def(py::init( [](Eigen::Ref<Eigen::VectorXd> const& b)
        {
            return new AffineFunction<Kokkos::HostSpace>(VecToKokkos<double, Kokkos::HostSpace>(b));
        }))
        .def(py::init( [](Eigen::Ref<Eigen::MatrixXd> const& A, Eigen::Ref<Eigen::VectorXd> const& b)
        {
            return new AffineFunction<Kokkos::HostSpace>(MatToKokkos<double, Kokkos::HostSpace>(A), VecToKokkos<double, Kokkos::HostSpace>(b));
        }))
        .def(py::init( [](Eigen::Ref<Eigen::MatrixXd> const& A)
        {
            return new AffineFunction<Kokkos::HostSpace>(MatToKokkos<double, Kokkos::HostSpace>(A));
        }))
        .def("print_Aij", &AffineFunction<Kokkos::HostSpace>::print_Aij)
        .def("print_ptr_A", &AffineFunction<Kokkos::HostSpace>::print_ptr_A);
}

#if defined(MPART_ENABLE_GPU)
void mpart::binding::AffineMapWrapperDevice(py::module &m)
{
    py::class_<AffineMap<mpart::DeviceSpace>, ConditionalMapBase<mpart::DeviceSpace>, std::shared_ptr<AffineMap<mpart::DeviceSpace>>>(m, "dAffineMap")
        .def(py::init( [](Eigen::Ref<Eigen::VectorXd> const& b)
        {
            return new AffineMap<mpart::DeviceSpace>(ToDevice<mpart::DeviceSpace>(Kokkos::View<double*,Kokkos::HostSpace>(VecToKokkos<double, Kokkos::HostSpace>(b))));
        }))
        .def(py::init( [](Eigen::Ref<Eigen::MatrixXd> const& A, Eigen::Ref<Eigen::VectorXd> const& b)
        {
            return new AffineMap<mpart::DeviceSpace>(ToDevice<mpart::DeviceSpace>(MatToKokkos<double, Kokkos::HostSpace>(A)), ToDevice<mpart::DeviceSpace>(Kokkos::View<double*,Kokkos::HostSpace>(VecToKokkos<double, Kokkos::HostSpace>(b))));
        }))
        .def(py::init( [](Eigen::Ref<Eigen::MatrixXd> const& A)
        {
            return new AffineMap<mpart::DeviceSpace>(ToDevice<mpart::DeviceSpace>(MatToKokkos<double, Kokkos::HostSpace>(A)));
        }));
}

void mpart::binding::AffineFunctionWrapperDevice(py::module &m)
{
    py::class_<AffineFunction<mpart::DeviceSpace>, ParameterizedFunctionBase<mpart::DeviceSpace>, std::shared_ptr<AffineFunction<mpart::DeviceSpace>>>(m, "dAffineFunction")
        .def(py::init( [](Eigen::Ref<Eigen::VectorXd> const& b)
        {
            return new AffineFunction<mpart::DeviceSpace>(ToDevice<mpart::DeviceSpace>(Kokkos::View<double*,Kokkos::HostSpace>(VecToKokkos<double, Kokkos::HostSpace>(b))));
        }))
        .def(py::init( [](Eigen::Ref<Eigen::MatrixXd> const& A, Eigen::Ref<Eigen::VectorXd> const& b)
        {
            return new AffineFunction<mpart::DeviceSpace>(ToDevice<mpart::DeviceSpace>(MatToKokkos<double, Kokkos::HostSpace>(A)), ToDevice<mpart::DeviceSpace>(Kokkos::View<double*,Kokkos::HostSpace>(VecToKokkos<double, Kokkos::HostSpace>(b))));
        }))
        .def(py::init( [](Eigen::Ref<Eigen::MatrixXd> const& A)
        {
            return new AffineFunction<mpart::DeviceSpace>(ToDevice<mpart::DeviceSpace>(MatToKokkos<double, Kokkos::HostSpace>(A)));
        }));
}
#endif // MPART_ENABLE_GPU