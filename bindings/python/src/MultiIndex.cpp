#include "CommonPybindUtilities.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexNeighborhood.h"
#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/MultiIndexLimiter.h"
#include "MParT/Utilities/ArrayConversions.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::MultiIndexWrapper(py::module &m)
{
    // MultiIndex
    py::class_<MultiIndex, KokkosCustomPointer<MultiIndex>>(m, "MultiIndex")
        .def(py::init<>())
        .def(py::init<unsigned int, unsigned int>())
        .def(py::init<std::vector<unsigned int> const&>())

        .def("sum", &MultiIndex::Sum)
        .def("max", &MultiIndex::Max)
        .def("count_nonzero", &MultiIndex::NumNz)
        .def("__setitem__", &MultiIndex::Set)
        .def("__getitem__", &MultiIndex::Get)
        .def("__str__", &MultiIndex::String)
        .def("__repl__",&MultiIndex::String)
        .def("__len__", &MultiIndex::Length)
        .def("__eq__", &MultiIndex::operator==)
        .def("__neq__", &MultiIndex::operator!=)
        .def("__lt__", &MultiIndex::operator<)
        .def("__gt__", &MultiIndex::operator>)
        .def("__le__", &MultiIndex::operator<=)
        .def("__ge__", &MultiIndex::operator>=);
        
    // ==========================================================================================================
    // MultiIndexSet
    py::class_<MultiIndexSet, KokkosCustomPointer<MultiIndexSet>>(m, "MultiIndexSet")

        .def(py::init<const unsigned int>())
        .def(py::init<Eigen::Ref<const Eigen::MatrixXi> const&>())

        //.def(py::init<>(Eigen::Matrix<unsigned int, Eigen::Dynamic, Eigen::Dynamic>))
        .def("fix", &MultiIndexSet::Fix)
        .def("__len__", &MultiIndexSet::Length)
        .def("access", &MultiIndexSet::operator[])
        .def("at", &MultiIndexSet::at)
        .def("Size", &MultiIndexSet::Size)
        //.def("append", &MultiIndexSet::operator+=)
        .def("union", &MultiIndexSet::Union)
        .def("SetLimiter",&MultiIndexSet::SetLimiter)
        //.def("GetLimiter", MultiIndexSet::GetLimiter)
        .def("IndexToMulti",&MultiIndexSet::IndexToMulti)
        .def("MultiToIndex", &MultiIndexSet::MultiToIndex)
        .def("MaxOrders", &MultiIndexSet::MaxOrders)
        //.def("Activate", &MultiIndexSet::Activate)
        .def("AddActivate", &MultiIndexSet::AddActive)
        ;
        

        // .def(py::init<unsigned int, unsigned int>())





    //==========================================================================================================
    //FixedMultiIndexSet

    py::class_<FixedMultiIndexSet<Kokkos::HostSpace>, KokkosCustomPointer<FixedMultiIndexSet<Kokkos::HostSpace>>>(m, "FixedMultiIndexSet")

        .def(py::init( [](unsigned int dim, 
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> &orders)
        {
            return new FixedMultiIndexSet<Kokkos::HostSpace>(dim, VecToKokkos<unsigned int>(orders));
        }))

        .def(py::init( [](unsigned int dim, 
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> &nzStarts,
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> &nzDims,
                          Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> &nzOrders)
        {
            return new FixedMultiIndexSet<Kokkos::HostSpace>(dim, 
                                          VecToKokkos<unsigned int>(nzStarts), 
                                          VecToKokkos<unsigned int>(nzDims), 
                                          VecToKokkos<unsigned int>(nzOrders));
        }))

        .def(py::init<unsigned int, unsigned int>())

        .def("MaxDegrees", [] (FixedMultiIndexSet<Kokkos::HostSpace> const &set)
        {
            auto maxDegrees = set.MaxDegrees(); // auto finds the type, but harder to read (because you don't tell reader the type)
            Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> maxDegreesEigen(maxDegrees.extent(0));
            for (int i = 0; i < maxDegrees.extent(0); i++)
            {
                maxDegreesEigen(i) = maxDegrees(i);
            }
            return maxDegreesEigen;
        })
        ;


}