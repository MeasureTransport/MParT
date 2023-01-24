#include "MParT/TrainMap.h"
#include "CommonJuliaUtilities.h"

void mpart::binding::TrainMapWrapper(jlcxx::Module &mod) {
    mod.add_type<TrainOptions>("__TrainOptions")
        .method("__opt_alg!", [](TrainOptions &opts, std::string alg){opts.opt_alg = alg;})
        .method("__opt_ftol_rel!", [](TrainOptions &opts, double tol){opts.opt_ftol_rel = tol;})
        .method("__opt_ftol_abs!", [](TrainOptions &opts, double tol){opts.opt_ftol_abs = tol;})
        .method("__opt_xtol_rel!", [](TrainOptions &opts, double tol){opts.opt_xtol_rel = tol;})
        .method("__opt_maxeval!", [](TrainOptions &opts, int eval){opts.opt_maxeval = eval;})
        .method("__verbose!", [](TrainOptions &opts, bool verbose){opts.verbose = verbose;})
    ;

    // TrainMap
    mod.method("TrainMap", &mpart::TrainMap<KLObjective<Kokkos::HostSpace>>);
}