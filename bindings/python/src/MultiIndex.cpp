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
        .def("HasNonzeroEnd", &MultiIndex::HasNonzeroEnd)

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
        .def("__len__", &MultiIndexSet::Length, "Retrieves the length of _each_ multiindex within this set (i.e. the dimension of the input)")
        .def("__getitem__", &MultiIndexSet::at)
        .def("at", &MultiIndexSet::at)
        .def("Size", &MultiIndexSet::Size, "Retrieves the number of elements in this MultiIndexSet")

        .def_static("CreateTotalOrder", &MultiIndexSet::CreateTotalOrder, py::arg("length"), py::arg("maxOrder"), py::arg("limiter")=MultiIndexLimiter::None())
        .def_static("CreateSeparableTotalOrder", [](unsigned int length, unsigned int maxOrder){return MultiIndexSet::CreateTotalOrder(length, maxOrder, MultiIndexLimiter::SeparableTotalOrder(maxOrder));}, py::arg("length"), py::arg("maxOrder"))
        .def_static("CreateTensorProduct", &MultiIndexSet::CreateTensorProduct, py::arg("length"), py::arg("maxOrder"), py::arg("limiter")=MultiIndexLimiter::None())

        .def("union", &MultiIndexSet::Union)
        .def("SetLimiter",&MultiIndexSet::SetLimiter)
        .def("GetLimiter", &MultiIndexSet::GetLimiter)
        .def("NonzeroDiagonalEntries", &MultiIndexSet::NonzeroDiagonalEntries)
        .def("IndexToMulti",&MultiIndexSet::IndexToMulti)
        .def("MultiToIndex", &MultiIndexSet::MultiToIndex)
        .def("MaxOrders", &MultiIndexSet::MaxOrders)
        .def("Expand", py::overload_cast<unsigned int>(&MultiIndexSet::Expand), "Expand frontier w.r.t one multiindex")
        .def("Expand", py::overload_cast<>(&MultiIndexSet::Expand), "Expand all frontiers of a MultiIndexSet")
        .def("append", py::overload_cast<MultiIndex const&>(&MultiIndexSet::operator+=))
        .def("__iadd__", py::overload_cast<MultiIndex const&>(&MultiIndexSet::operator+=))
        .def("DeepCopy",[](const MultiIndexSet& mset)
        {
            MultiIndexSet mset_copy = mset;
            return mset_copy;
        })
        .def("Activate", py::overload_cast<MultiIndex const&>(&MultiIndexSet::Activate))
        .def("AddActive", &MultiIndexSet::AddActive)
        .def("Frontier", &MultiIndexSet::Frontier)
        .def("Margin", &MultiIndexSet::Margin)
        .def("ReducedMargin", &MultiIndexSet::ReducedMargin)
        .def("ReducedMarginDim", &MultiIndexSet::ReducedMarginDim)
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
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> nzStartsIn,
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> nzDimsIn,
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> nzOrdersIn)
        {   
            // Deep copy the arrays into Kokkos 
            Kokkos::View<unsigned int*,Kokkos::HostSpace> nzStarts("nzStarts", nzStartsIn.rows());
            Kokkos::View<unsigned int*,Kokkos::HostSpace> nzDims("nzDims", nzDimsIn.rows());
            Kokkos::View<unsigned int*,Kokkos::HostSpace> nzOrders("nzOrders", nzOrdersIn.rows());

            for(unsigned int i=0; i<nzStartsIn.rows(); ++i)
                nzStarts(i) = nzStartsIn(i);
            for(unsigned int i=0; i<nzDimsIn.rows(); ++i)
                nzDims(i) = nzDimsIn(i);
            for(unsigned int i=0; i<nzOrdersIn.rows(); ++i)
                nzOrders(i) = nzOrdersIn(i);
                
            return new FixedMultiIndexSet<Kokkos::HostSpace>(dim, nzStarts, nzDims, nzOrders);
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
        .def("__len__", &FixedMultiIndexSet<Kokkos::HostSpace>::Length)
        .def("Length", &FixedMultiIndexSet<Kokkos::HostSpace>::Length)
        .def("Size", &FixedMultiIndexSet<Kokkos::HostSpace>::Size)
        .def("IndexToMulti", &FixedMultiIndexSet<Kokkos::HostSpace>::IndexToMulti)
        .def("MultiToIndex", &FixedMultiIndexSet<Kokkos::HostSpace>::MultiToIndex)

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
        .def(py::pickle(
            [](FixedMultiIndexSet<Kokkos::HostSpace> &mset) { // __getstate__
                std::stringstream ss;
                cereal::BinaryOutputArchive oarchive(ss);
                oarchive(mset);
                return py::bytes(ss.str());
            },
            [](py::bytes input) {
                
                FixedMultiIndexSet<Kokkos::HostSpace> mset;

                std::stringstream ss;
                ss.str(input);
                cereal::BinaryInputArchive iarchive(ss);
                iarchive(mset);

                return mset;
            }
        ))
#endif // MPART_HAS_CEREAL
#if defined(MPART_ENABLE_GPU)
        .def("ToDevice", &FixedMultiIndexSet<Kokkos::HostSpace>::ToDevice<mpart::DeviceSpace>)
        ;
    py::class_<FixedMultiIndexSet<mpart::DeviceSpace>, std::shared_ptr<FixedMultiIndexSet<mpart::DeviceSpace>>>(m, "dFixedMultiIndexSet")
#endif // MPART_ENABLE_GPU
    ;
}