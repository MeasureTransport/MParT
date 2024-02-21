#include <fstream>
#include "MParT/MapOptions.h"
#include <Kokkos_Core.hpp>

#if defined(MPART_HAS_CEREAL)
#include "MParT/Utilities/Serialization.h"
#endif // MPART_HAS_CEREAL

#include "CommonJuliaUtilities.h"

using namespace mpart;

void mpart::binding::MapOptionsWrapper(jlcxx::Module &mod) {
    // BasisTypes
    mod.add_bits<BasisTypes>("__BasisTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("__ProbabilistHermite", BasisTypes::ProbabilistHermite);
    mod.set_const("__PhysicistHermite", BasisTypes::PhysicistHermite);
    mod.set_const("__HermiteFunctions", BasisTypes::HermiteFunctions);

    // PosFuncTypes
    mod.add_bits<PosFuncTypes>("__PosFuncTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("__Exp", PosFuncTypes::Exp);
    mod.set_const("__SoftPlus", PosFuncTypes::SoftPlus);

    // QuadTypes
    mod.add_bits<QuadTypes>("__QuadTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("__ClenshawCurtis", QuadTypes::ClenshawCurtis);
    mod.set_const("__AdaptiveSimpson", QuadTypes::AdaptiveSimpson);
    mod.set_const("__AdaptiveClenshawCurtis", QuadTypes::AdaptiveClenshawCurtis);

    // SigmoidTypes
    mod.add_bits<SigmoidTypes>("__SigmoidTypes", jlcxx::julia_type("CppEnum"));
    mod.set_const("__Logistic", SigmoidTypes::Logistic);

    // EdgeTypes: TODO: SoftPlus overlaps with PosFuncTypes, needs to be fixed
    // mod.add_bits<EdgeTypes>("__EdgeTypes", jlcxx::julia_type("CppEnum"));
    // mod.set_const("__SoftPlus", EdgeTypes::SoftPlus);

    // MapOptions
    mod.add_type<MapOptions>("__MapOptions")
        .method("__basisType!", [](MapOptions &opts, unsigned int basis){ opts.basisType = static_cast<BasisTypes>(basis); })
        .method("__basisLB!", [](MapOptions &opts, double lb){ opts.basisLB = lb; })
        .method("__basisUB!", [](MapOptions &opts, double ub){ opts.basisUB = ub; })
        .method("__basisNorm!", [](MapOptions &opts, bool shouldNorm){ opts.basisNorm = shouldNorm; })
        .method("__posFuncType!", [](MapOptions &opts, unsigned int f){ opts.posFuncType = static_cast<PosFuncTypes>(f); })
        .method("__quadType!", [](MapOptions &opts, unsigned int quad){ opts.quadType = static_cast<QuadTypes>(quad); })
        .method("__sigmoidType!", [](MapOptions &opts, unsigned int sig){ opts.sigmoidType = static_cast<SigmoidTypes>(sig); })
        .method("__edgeType!", [](MapOptions &opts, unsigned int edge){ opts.edgeType = static_cast<EdgeTypes>(edge); })
        .method("__edgeShape!", [](MapOptions &opts, double width){ opts.edgeShape = width; })
        .method("__quadAbsTol!", [](MapOptions &opts, double tol){ opts.quadAbsTol = tol; })
        .method("__quadRelTol!", [](MapOptions &opts, double tol){ opts.quadRelTol = tol; })
        .method("__quadMaxSub!", [](MapOptions &opts, unsigned int sub){ opts.quadMaxSub = sub; })
        .method("__quadMinSub!", [](MapOptions &opts, unsigned int sub){ opts.quadMinSub = sub; })
        .method("__quadPts!", [](MapOptions &opts, unsigned int pts){ opts.quadPts = pts; })
        .method("__contDeriv!", [](MapOptions &opts, bool deriv){ opts.contDeriv = deriv; })
        .method("__nugget!", [](MapOptions &opts, double nugget){ opts.nugget = nugget; })
        .method("Serialize", [](MapOptions &opts, std::string &filename){
#if defined(MPART_HAS_CEREAL)
            std::ofstream os (filename);
            cereal::BinaryOutputArchive oarchive(os);
            oarchive(opts);
#else
            std::cerr << "MapOptions::Serialize: MParT was not compiled with Cereal support. Operation incomplete." << std::endl;
#endif
        })
        .method("Deserialize", [](MapOptions &opts, std::string &filename){
#if defined(MPART_HAS_CEREAL)
            std::ifstream is (filename);
            cereal::BinaryInputArchive iarchive(is);
            iarchive(opts);
#else
            std::cerr << "MapOptions::Deserialize: MParT was not compiled with Cereal support. Operation incomplete." << std::endl;
#endif
        })
    ;

    mod.set_override_module(jl_base_module);
    mod.method("string", [](MapOptions opts){return opts.String();});
    mod.method("==", [](MapOptions opts1, MapOptions opts2){return opts1 == opts2;});
    mod.unset_override_module();
}
