#include "MParT/TrainMap.h"
#include "CommonJuliaUtilities.h"

void mpart::binding::TrainMapWrapper(jlcxx::Module &mod) {
    mod.add_type<TrainOptions>("__TrainOptions")
        .method("__opt_alg!", [](TrainOptions &opts, std::string alg){opts.opt_alg = alg;})
        .method("__opt_ftol_rel!", [](TrainOptions &opts, double tol){opts.opt_ftol_rel = tol;})
        .method("__opt_ftol_abs!", [](TrainOptions &opts, double tol){opts.opt_ftol_abs = tol;})
        .method("__opt_xtol_rel!", [](TrainOptions &opts, double tol){opts.opt_xtol_rel = tol;})
        .method("__opt_xtol_abs!", [](TrainOptions &opts, double tol){opts.opt_xtol_abs = tol;})
        .method("__opt_maxeval!", [](TrainOptions &opts, int eval){opts.opt_maxeval = eval;})
        .method("__verbose!", [](TrainOptions &opts, bool verbose){opts.verbose = verbose;})
    ;

    mod.set_override_module(jl_base_module);
    mod.method("string", [](const TrainOptions &opts){
        std::stringstream ss;
        ss << "opt_alg = " << opts.opt_alg << "\n";
        ss << "opt_ftol_rel = " << opts.opt_ftol_rel << "\n";
        ss << "opt_ftol_abs = " << opts.opt_ftol_abs << "\n";
        ss << "opt_xtol_rel = " << opts.opt_xtol_rel << "\n";
        ss << "opt_xtol_abs = " << opts.opt_xtol_abs << "\n";
        ss << "opt_maxeval = " << opts.opt_maxeval << "\n";
        ss << "verbose = " << (opts.verbose ? "true" : "false") << "\n";
        return ss.str();
    });
    mod.unset_override_module();

    // TrainMap
    mod.method("TrainMap", &mpart::TrainMap<KLObjective<Kokkos::HostSpace>>);
}