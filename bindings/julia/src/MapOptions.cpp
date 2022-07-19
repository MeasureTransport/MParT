#include <MParT/MapOptions.h>
#include <Kokkos_Core.hpp>

#include "CommonJuliaUtilities.h"

using namespace mpart::binding;

void mpart::binding::MapOptionsWrapper(jlcxx::module &m) {
    // BasisTypes
    m.add_bits<BasisTypes>("BasisTypes", jlcxx::julia_type("CppEnum"));
    m.value("ProbabilistHermite", BasisTypes::ProbabilistHermite);
    m.value("PhysicistHermite", BasisTypes::PhysicistHermite);
    m.value("HermiteFunctions", BasisTypes::HermiteFunctions);

    // PosFuncTypes
    m.add_bits<PosFuncTypes>("PosFuncTypes", jlcxx::julia_type("CppEnum"));
    m.value("Exp", PosFuncTypes::Exp);
    m.value("SoftPlus", PosFuncTypes::SoftPlus);

    // QuadTypes
    m.add_bits<QuadTypes>("QuadTypes", jlcxx::julia_type("CppEnum"));
    m.value("ClenshawCurtis", QuadTypes::ClenshawCurtis);
    m.value("AdaptiveSimpson", QuadTypes::AdaptiveSimpson);
    m.value("AdaptiveClenshawCurtis", QuadTypes::AdaptiveClenshawCurtis);

    // MapOptions
    m.add_type<MapOptions>("MapOptions").constructor<>();
}