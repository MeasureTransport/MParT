#include "CommonJuliaUtilities.h"
#include "MParT/MapOptions.h"
#include "MParT/MapObjective.h"
#include "MParT/TrainMap.h"
#include "MParT/AdaptiveTransportMap.h"

#include <Kokkos_Core.hpp>

using namespace mpart;

namespace jlcxx {
    // Tell CxxWrap.jl the supertype structure for ConditionalMapBase
    template<> struct SuperType<mpart::ATMOptions> {typedef mpart::MapOptions type;};
}

void mpart::binding::AdaptiveTransportMapWrapper(jlcxx::Module &mod) {
    // Can only do single inheritence, so I arbitrarily picked inheritence from MapOptions
    // If you need to convert to TrainOptions, I allow a conversion
    mod.add_type<ATMOptions>("__ATMOptions", jlcxx::julia_base_type<MapOptions>())
        .method("__opt_alg!", [](ATMOptions &opts, std::string alg){opts.opt_alg = alg;})
        .method("__opt_ftol_rel!", [](ATMOptions &opts, double tol){opts.opt_ftol_rel = tol;})
        .method("__opt_ftol_abs!", [](ATMOptions &opts, double tol){opts.opt_ftol_abs = tol;})
        .method("__opt_xtol_rel!", [](ATMOptions &opts, double tol){opts.opt_xtol_rel = tol;})
        .method("__opt_xtol_abs!", [](ATMOptions &opts, double tol){opts.opt_xtol_abs = tol;})
        .method("__opt_maxeval!", [](ATMOptions &opts, int eval){opts.opt_maxeval = eval;})
        .method("__verbose!", [](ATMOptions &opts, int verbose){opts.verbose = verbose;})
        .method("__maxPatience!", [](ATMOptions &opts, int maxPatience){opts.maxPatience = maxPatience;})
        .method("__maxSize!", [](ATMOptions &opts, int maxSize){opts.maxSize = maxSize;})
        .method("__maxDegrees!", [](ATMOptions &opts, MultiIndex &maxDegrees){opts.maxDegrees = maxDegrees;})
        .method("TrainOptions", [](ATMOptions &opts){ return static_cast<TrainOptions>(opts);})
    ;


    mod.method("AdaptiveTransportMap", [](jlcxx::ArrayRef<MultiIndexSet> arr, std::shared_ptr<MapObjective<Kokkos::HostSpace>> objective, ATMOptions options) {
        std::vector<MultiIndexSet> vec (arr.begin(), arr.end());
        auto map = AdaptiveTransportMap(vec, objective, options);
        for(int i = 0; i < vec.size(); i++) arr[i] = vec[i];
        return map;
    });
}
