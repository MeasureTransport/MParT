#include <MParT/MapOptions.h>
#include <Kokkos_Core.hpp>

#include "CommonJuliaUtilities.h"

using namespace mpart::binding;

void mpart::binding::MapOptionsWrapper(jlcxx::module &m) {
    // BasisTypes
    mod.add_bits<BasisTypes>("BasisTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("ProbabilistHermite", BasisTypes::ProbabilistHermite);
    mod.set_const("PhysicistHermite", BasisTypes::PhysicistHermite);
    mod.set_const("HermiteFunctions", BasisTypes::HermiteFunctions);

    // PosFuncTypes
    mod.add_bits<PosFuncTypes>("PosFuncTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("Exp", PosFuncTypes::Exp);
    mod.set_const("SoftPlus", PosFuncTypes::SoftPlus);

    // QuadTypes
    mod.add_bits<QuadTypes>("QuadTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("ClenshawCurtis", QuadTypes::ClenshawCurtis);
    mod.set_const("AdaptiveSimpson", QuadTypes::AdaptiveSimpson);
    mod.set_const("AdaptiveClenshawCurtis", QuadTypes::AdaptiveClenshawCurtis);

    // MapOptions
    mod.add_type<MapOptions>("MapOptions").constructor<>()
        .method("BasisType!", [](MapOptions &opts, BasisTypes basis){ opts.basisType = basis; })
        .method("PosFuncType!", [](MapOptions &opts, PosFuncTypes f){ opts.posFuncType = f; })
        .method("QuadType!", [](MapOptions &opts, QuadTypes quad){ opts.quadType = quad; })
        .method("QuadAbsTol!", [](MapOptions &opts, double tol){ opts.quadAbsTol = tol; })
        .method("QuadRelTol!", [](MapOptions &opts, double tol){ opts.quadRelTol = tol; })
        .method("QuadMaxSub!", [](MapOptions &opts, unsigned int sub){ opts.quadMaxSub = sub; })
        .method("QuadMinSub!", [](MapOptions &opts, unsigned int sub){ opts.quadMinSub = sub; })
        .method("QuadPts!", [](MapOptions &opts, unsigned int pts){ opts.quadPts = pts; })
        .method("ContDeriv!", [](MapOptions &opts, bool deriv){ opts.contDeriv = deriv; })
        ;
}