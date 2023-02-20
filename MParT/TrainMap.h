#ifndef MPART_TRAINMAP_H
#define MPART_TRAINMAP_H

#include <functional>
#include <nlopt.hpp>
#include "ConditionalMapBase.h"
#include "MapObjective.h"

namespace mpart {
/**
 * @brief Concise struct representing options for optimizing a map using NLopt
 *
 */
struct TrainOptions {
    std::string opt_alg = "LD_LBFGS";
    double opt_stopval = -std::numeric_limits<double>::infinity();
    double opt_ftol_rel = 1e-3;
    double opt_ftol_abs = 1e-3;
    double opt_xtol_rel = 1e-4;
    double opt_xtol_abs = 1e-4;
    int opt_maxeval = 30;
    double opt_maxtime = 100.;
    bool verbose = false;
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
        ss << "verbose = " << (verbose ? "true" : "false");
        return ss.str();
    }
};

/**
 * @brief Function to train a map inplace given an objective and optimization options
 *
 * @tparam ObjectiveType
 * @param map Map to optimize (inplace)
 * @param objective MapObjective to optimize over
 * @param options Options for optimizing the map
 */
template<typename ObjectiveType>
void TrainMap(std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map, ObjectiveType &objective, TrainOptions options);

} // namespace mpart

#endif // MPART_TRAINMAP_H