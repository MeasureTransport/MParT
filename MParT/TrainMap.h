#ifndef MPART_TRAINMAP_H
#define MPART_TRAINMAP_H

#include <nlopt.hpp>
#include "ConditionalMapBase.h"
#include "MapObjective.h"

namespace mpart {
struct TrainOptions {
    std::string opt_alg = "LD_LBFGS";
    double opt_stopval = -std::numeric_limits<double>::infinity();
    double opt_ftol_rel = 1e-3;
    double opt_ftol_abs = 1e-3;
    double opt_xtol_rel = 1e-4;
    int opt_maxeval = 10;
    double opt_maxtime = 100.;
    bool verbose = false;
};

void TrainMap(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<MapObjective<MemorySpace>> objective, TrainOptions options = TrainOptions());

} // namespace mpart

#endif // MPART_TRAINMAP_H