#include "CommonPybindUtilities.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexNeighborhood.h"
#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/MultiIndexLimiter.h"
#include "MParT/Utilities/ArrayConversions.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/functional.h>

#include <Kokkos_Core.hpp>

#include <pybind11/pybind11.h>


#if defined(MPART_HAS_CEREAL)
#include "MParT/Utilities/Serialization.h"
#include <fstream>
#endif // MPART_HAS_CEREAL


namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::MultiIndexWrapper(py::module &m)
{
    // MultiIndexSetLimiters
    //TotalOrder
    py::class_<MultiIndexLimiter::TotalOrder, std::shared_ptr<MultiIndexLimiter::TotalOrder>>(m, "TotalOrder")
        .def(py::init<unsigned int>())
        .def("__call__", &MultiIndexLimiter::TotalOrder::operator())
    ;


    //Dimension
    py::class_<MultiIndexLimiter::Dimension, std::shared_ptr<MultiIndexLimiter::Dimension>>(m, "Dimension")
        .def(py::init<unsigned int, unsigned int>())
        .def("__call__", &MultiIndexLimiter::Dimension::operator())
    ;


    //Anisotropic
    py::class_<MultiIndexLimiter::Anisotropic, std::shared_ptr<MultiIndexLimiter::Anisotropic>>(m, "Anisotropic")
        .def(py::init<std::vector<double> const&, double>())
        .def("__call__", &MultiIndexLimiter::Anisotropic::operator())
    ;


    //MaxDegree
    py::class_<MultiIndexLimiter::MaxDegree, std::shared_ptr<MultiIndexLimiter::MaxDegree>>(m, "MaxDegree")
        .def(py::init<unsigned int, unsigned int>())
        .def("__call__", &MultiIndexLimiter::MaxDegree::operator())
    ;


    //None
    py::class_<MultiIndexLimiter::None, std::shared_ptr<MultiIndexLimiter::None>>(m, "NoneLim")
        .def(py::init<>())
        .def("__call__", &MultiIndexLimiter::None::operator())
    ;


    //And
    py::class_<MultiIndexLimiter::And, std::shared_ptr<MultiIndexLimiter::And>>(m, "And")
        .def(py::init<std::function<bool(MultiIndex const&)>,std::function<bool(MultiIndex const&)>>())
        .def("__call__", &MultiIndexLimiter::And::operator())
    ;

    //Or
    py::class_<MultiIndexLimiter::Or, std::shared_ptr<MultiIndexLimiter::Or>>(m, "Or")
        .def(py::init<std::function<bool(MultiIndex const&)>,std::function<bool(MultiIndex const&)>>())
        .def("__call__", &MultiIndexLimiter::Or::operator())
    ;

    //Xor
    py::class_<MultiIndexLimiter::Xor, std::shared_ptr<MultiIndexLimiter::Xor>>(m, "Xor")
        .def(py::init<std::function<bool(MultiIndex const&)>,std::function<bool(MultiIndex const&)>>())
        .def("__call__", &MultiIndexLimiter::Xor::operator())
    ;


    // MultiIndex
    py::class_<MultiIndex, std::shared_ptr<MultiIndex>>(m, "MultiIndex")
        .def(py::init<>())
        .def(py::init<unsigned int>())
        .def(py::init<unsigned int, unsigned int>())
        .def(py::init<std::vector<unsigned int> const&>())

        .def("tolist",&MultiIndex::Vector)
        .def("sum", &MultiIndex::Sum)
        .def("max", &MultiIndex::Max)

        .def("__setitem__", &MultiIndex::Set)
        .def("__getitem__", &MultiIndex::Get)
        .def("count_nonzero", &MultiIndex::NumNz)

        .def("__str__", &MultiIndex::String)
        .def("__repl__",&MultiIndex::String)
        .def("__len__", &MultiIndex::Length)
        .def("__eq__", &MultiIndex::operator==)
        .def("__neq__", &MultiIndex::operator!=)
        .def("__lt__", &MultiIndex::operator<)
        .def("__gt__", &MultiIndex::operator>)
        .def("__le__", &MultiIndex::operator<=)
        .def("__ge__", &MultiIndex::operator>=);


    // MultiIndexSet
    py::class_<MultiIndexSet, std::shared_ptr<MultiIndexSet>>(m, "MultiIndexSet")
        .def(py::init<const unsigned int>())
        .def(py::init<Eigen::Ref<const Eigen::MatrixXi> const&>())
        .def("fix", &MultiIndexSet::Fix)
        .def("__len__", &MultiIndexSet::Length)
        .def("__getitem__", &MultiIndexSet::at)
        .def("at", &MultiIndexSet::at)
        .def("Size", &MultiIndexSet::Size)

        .def_static("CreateTotalOrder", &MultiIndexSet::CreateTotalOrder, py::arg("length"), py::arg("maxOrder"), py::arg("limiter")=MultiIndexLimiter::None())
        .def_static("CreateTensorProduct", &MultiIndexSet::CreateTensorProduct, py::arg("length"), py::arg("maxOrder"), py::arg("limiter")=MultiIndexLimiter::None())

        .def("union", &MultiIndexSet::Union)
        .def("SetLimiter",&MultiIndexSet::SetLimiter)
        .def("GetLimiter", &MultiIndexSet::GetLimiter)
        .def("IndexToMulti",&MultiIndexSet::IndexToMulti)
        .def("MultiToIndex", &MultiIndexSet::MultiToIndex)
        .def("MaxOrders", &MultiIndexSet::MaxOrders)
        .def("Expand", py::overload_cast<unsigned int>(&MultiIndexSet::Expand))
        .def("append", py::overload_cast<MultiIndex const&>(&MultiIndexSet::operator+=))
        .def("__iadd__", py::overload_cast<MultiIndex const&>(&MultiIndexSet::operator+=))
        .def("Activate", py::overload_cast<MultiIndex const&>(&MultiIndexSet::Activate))
        .def("AddActive", &MultiIndexSet::AddActive)
        .def("Frontier", &MultiIndexSet::Frontier)
        .def("Margin", &MultiIndexSet::Margin)
        .def("ReducedMargin", &MultiIndexSet::ReducedMargin)
        .def("StrictFrontier", &MultiIndexSet::StrictFrontier)
        .def("IsExpandable", &MultiIndexSet::IsExpandable)
        .def("NumActiveForward", &MultiIndexSet::NumActiveForward)
        .def("NumForward", &MultiIndexSet::NumForward)
        //.def("IsAdmissible", &MultiIndexSet::IsAdmissible)
        //.def("IsActive", &MultiIndexSet::IsActive)
        //.def("Visualize", &MultiIndexSet::Visualize)
        ;



    //==========================================================================================================
    //FixedMultiIndexSet

    py::class_<FixedMultiIndexSet<Kokkos::HostSpace>, std::shared_ptr<FixedMultiIndexSet<Kokkos::HostSpace>>>(m, "FixedMultiIndexSet")

        .def(py::init( [](unsigned int dim,
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> &orders)
        {
            return new FixedMultiIndexSet<Kokkos::HostSpace>(dim, VecToKokkos<unsigned int,Kokkos::HostSpace>(orders));
        }))

        .def(py::init( [](unsigned int dim,
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> &nzStarts,
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> &nzDims,
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> &nzOrders)
        {
            return new FixedMultiIndexSet<Kokkos::HostSpace>(dim,
                                          VecToKokkos<unsigned int, Kokkos::HostSpace>(nzStarts),
                                          VecToKokkos<unsigned int, Kokkos::HostSpace>(nzDims),
                                          VecToKokkos<unsigned int, Kokkos::HostSpace>(nzOrders));
        }))

        .def(py::init<unsigned int, unsigned int>())

        .def("MaxDegrees", [] (const FixedMultiIndexSet<Kokkos::HostSpace> &set)
        {
            auto maxDegrees = set.MaxDegrees();
            Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> maxDegreesEigen(maxDegrees.extent(0));
            for (unsigned int i = 0; i < maxDegrees.extent(0); i++)
            {
                maxDegreesEigen(i) = maxDegrees(i);
            }
            return maxDegreesEigen;
        })

#if defined(MPART_HAS_CEREAL)
        .def("Serialize", [](FixedMultiIndexSet<Kokkos::HostSpace> const &mset, std::string const &filename){
            std::ofstream os(filename);
            cereal::BinaryOutputArchive oarchive(os);
            oarchive(mset);
            return mset;
        })
        .def("Deserialize", [](FixedMultiIndexSet<Kokkos::HostSpace> &mset, std::string const &filename){
            std::ifstream is(filename);
            cereal::BinaryInputArchive iarchive(is);
            iarchive(mset);
            return mset;
        })
#endif // MPART_HAS_CEREAL
#if defined(MPART_ENABLE_GPU)
        .def("ToDevice", &FixedMultiIndexSet<Kokkos::HostSpace>::ToDevice<mpart::DeviceSpace>)
        ;
    py::class_<FixedMultiIndexSet<mpart::DeviceSpace>, std::shared_ptr<FixedMultiIndexSet<mpart::DeviceSpace>>>(m, "dFixedMultiIndexSet")
#endif // MPART_ENABLE_GPU
    ;
}