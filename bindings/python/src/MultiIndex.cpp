#include "CommonPybindUtilities.h"
#include "MParT/MultiIndices/MultiIndex.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::python;

void mpart::python::MultiIndexWrapper(py::module &m)
{
    py::class_<MultiIndex, KokkosCustomPointer<MultiIndex>>(m, "MultiIndex")
        .def(py::init<>())
        .def(py::init<unsigned int, unsigned int>())
        .def(py::init<std::vector<unsigned int> const&>())

        .def("Sum", &MultiIndex::Sum)
        .def("Max", &MultiIndex::Max)
        .def("Set", &MultiIndex::Set)
        .def("Get", &MultiIndex::Get)
        .def("NumNz", &MultiIndex::NumNz)
        .def("String", &MultiIndex::String)
        .def("Length", &MultiIndex::Length)
        .def("operator==", &MultiIndex::operator==)
        .def("operator!=", &MultiIndex::operator!=)
        .def("operator<", &MultiIndex::operator<)
        .def("operator>", &MultiIndex::operator>)
        .def("operator<=", &MultiIndex::operator<=)
        .def("operator>=", &MultiIndex::operator>=);

    // // Will be used for MultiIndexSet
    // py::class_<MultiIndexSet, KokkosCustomPointer<MultiIndex>>(m, "MultiIndex")
    //     .def(py::init<>())
    //     .def(py::init<unsigned int, unsigned int>())
    //     .def(py::init<std::vector<unsigned int> const&>())

    //     .def("Sum", &MultiIndex::Sum)
    //     .def("Max", &MultiIndex::Max)
    //     .def("Set", &MultiIndex::Set);
    
}