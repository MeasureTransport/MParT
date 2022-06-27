#include "CommonJuliaUtilities.h"
#include "MParT/MapOptions.h"

#include <Kokkos_Core.hpp>

namespace py = pybind11;
using namespace mpart::binding;

JLCXX_MODULE mpart::binding::MapOptionsWrapper(jlcxx::Module& types)
{

    // BasisTypes
    types.add_bits<BasisTypes>("BasisTypes", jlcxx::julia_type("CppEnum"));
    types.set_const("ProbabilistHermite", BasisTypes::ProbabilistHermite);
    types.set_const("PhysicistHermite",   BasisTypes::PhysicistHermite);
    types.set_const("HermiteFunctions",   BasisTypes::HermiteFunctions);

    // PosFuncTypes
    types.add_bits<PosFuncTypes>("PosFuncTypes", jlcxx::julia_type("CppEnum"));
    types.set_const("Exp", PosFuncTypes::Exp);
    types.set_const("SoftPlus", PosFuncTypes::SoftPlus);

    // QuadTypes
    types.add_bits<QuadTypes>("QuadTypes", jlcxx::julia_type("CppEnum"));
    types.set_const("ClenshawCurtis", QuadTypes::ClenshawCurtis);
    types.set_const("AdaptiveSimpson", QuadTypes::AdaptiveSimpson);
    types.set_const("AdaptiveClenshawCurtis", QuadTypes::AdaptiveClenshawCurtis);

    // MapOptions
    py::class_<MapOptions, KokkosCustomPointer<MapOptions>>(m, "MapOptions")
    .def(py::init<>())
    .def_readwrite("basisType", &MapOptions::basisType)
    .def_readwrite("posFuncType", &MapOptions::posFuncType)
    .def_readwrite("quadType", &MapOptions::quadType)
    .def_readwrite("quadAbsTol", &MapOptions::quadAbsTol)
    .def_readwrite("quadRelTol", &MapOptions::quadRelTol)
    .def_readwrite("quadMaxSub", &MapOptions::quadMaxSub)
    .def_readwrite("quadPts", &MapOptions::quadPts);
}