#include <MParT/MapOptions.h>
#include <Kokkos_Core.hpp>

#include "CommonJuliaUtilities.h"

using namespace mpart;

JLCXX_MODULE BasisType_julia_module(jlcxx::Module& mod) {
    // BasisTypes
    mod.add_bits<BasisTypes>("__BasisTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("ProbabilistHermite", BasisTypes::ProbabilistHermite);
    mod.set_const("PhysicistHermite", BasisTypes::PhysicistHermite);
    mod.set_const("HermiteFunctions", BasisTypes::HermiteFunctions);
}

JLCXX_MODULE PosFuncType_julia_module(jlcxx::Module &mod) {
    // PosFuncTypes
    mod.add_bits<PosFuncTypes>("__PosFuncTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("Exp", PosFuncTypes::Exp);
    mod.set_const("SoftPlus", PosFuncTypes::SoftPlus);
}

JLCXX_MODULE QuadType_julia_module(jlcxx::Module &mod) {
    // QuadTypes
    mod.add_bits<QuadTypes>("__QuadTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("ClenshawCurtis", QuadTypes::ClenshawCurtis);
    mod.set_const("AdaptiveSimpson", QuadTypes::AdaptiveSimpson);
    mod.set_const("AdaptiveClenshawCurtis", QuadTypes::AdaptiveClenshawCurtis);
}

void mpart::binding::MapOptionsWrapper(jlcxx::Module &mod) {
    // MapOptions
    mod.add_type<MapOptions>("__MapOptions")
        .method("__basisType!", [](MapOptions &opts, unsigned int basis){ opts.basisType = static_cast<BasisTypes>(basis); })
        .method("__basisLB!", [](MapOptions &opts, double lb){ opts.basisLB = lb; })
        .method("__basisUB!", [](MapOptions &opts, double ub){ opts.basisUB = ub; })
        .method("__posFuncType!", [](MapOptions &opts, unsigned int f){ opts.posFuncType = static_cast<PosFuncTypes>(f); })
        .method("__quadType!", [](MapOptions &opts, unsigned int quad){ opts.quadType = static_cast<QuadTypes>(quad); })
        .method("__quadAbsTol!", [](MapOptions &opts, double tol){ opts.quadAbsTol = tol; })
        .method("__quadRelTol!", [](MapOptions &opts, double tol){ opts.quadRelTol = tol; })
        .method("__quadMaxSub!", [](MapOptions &opts, unsigned int sub){ opts.quadMaxSub = sub; })
        .method("__quadMinSub!", [](MapOptions &opts, unsigned int sub){ opts.quadMinSub = sub; })
        .method("__quadPts!", [](MapOptions &opts, unsigned int pts){ opts.quadPts = pts; })
        .method("__contDeriv!", [](MapOptions &opts, bool deriv){ opts.contDeriv = deriv; })
    ;
}