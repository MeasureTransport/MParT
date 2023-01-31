#ifndef MPART_TRAINMAP_H
#define MPART_TRAINMAP_H

#include <functional>
#include <nlopt.hpp>
#include "ConditionalMapBase.h"
#include "MapObjective.h"

namespace mpart {

/**
 * @brief TrainOptions adds options for training your map,
 * with fields largely based on nlopt settings. verbose is an integer
 * where 0=nothing, 1=some diagnostics, 2=debugging
 *
 */
struct TrainOptions {
    std::string opt_alg = "LD_SLSQP";
    double opt_stopval = -std::numeric_limits<double>::infinity();
    double opt_ftol_rel = 1e-3;
    double opt_ftol_abs = 1e-3;
    double opt_xtol_rel = 1e-4;
    double opt_xtol_abs = 1e-4;
    int opt_maxeval = 30;
    double opt_maxtime = 100.;
    int verbose = 0;
    std::string String() {
        std::stringstream ss;
        ss << "opt_alg = " << opt_alg << "\n";
        ss << "opt_stopval = " << opt_stopval << "\n";
        ss << "opt_ftol_rel = " << opt_ftol_rel << "\n";
        ss << "opt_ftol_abs = " << opt_ftol_abs << "\n";
        ss << "opt_xtol_rel = " << opt_xtol_rel << "\n";
        ss << "opt_xtol_abs = " << opt_xtol_abs << "\n";
        ss << "opt_maxeval = " << opt_maxeval << "\n";
        ss << "opt_maxtime = " << opt_maxtime << "\n";
        ss << "verbose = " << verbose;
        return ss.str();
    }
};

template<typename MemorySpace>
double TrainMap(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<MapObjective<MemorySpace>> objective, TrainOptions options);

} // namespace mpart

#endif // MPART_TRAINMAP_H