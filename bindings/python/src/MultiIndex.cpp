#include "CommonPybindUtilities.h"
#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
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
        


    // FixedMultiIndexSet
    py::class_<FixedMultiIndexSet, KokkosCustomPointer<FixedMultiIndexSet>>(m, "FixedMultiIndex")

        .def(py::init( [](unsigned int dim, Eigen::Matrix<unsigned int, Eigen::Dynamic, 1> const& orders){
            return new FixedMultiIndexSet(dim, VecToKokkos<unsigned int>(orders));
            }
            )
            );

}