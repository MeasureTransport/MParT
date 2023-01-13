#include "MParT/TrainMap.h"


nlopt::opt SetupOptimization(unsigned int dim, TrainOptions options) {
    nlopt::opt opt(options.opt_alg.c_str(), dim);
    // TODO: Set all the optimization options here
    opt.set_stopval(options.opt_stopval);
    opt.set_xtol_rel(options.opt_xtol_rel);
    opt.set_ftol_rel(options.opt_ftol_rel);
    opt.set_ftol_abs(options.opt_ftol_abs);
    opt.set_maxeval(options.opt_maxeval);
    opt.set_maxtime(options.opt_maxtime);

    // TODO: Print out all of the optimization settings here if verbose
    if(options.verbose){
        std::cout << "Optimization Settings:\n";
        std::cout << "Algorithm: " << opt.get_algorithm_name() << "\n";
        std::cout << "Optimization dimension:" << opt.get_dimension() << "\n";
        std::cout << "Optimization stopval:" << opt.get_stopval() << "\n";
        std::cout << "Max f evaluations: " << opt.get_maxeval() << "\n";
        std::cout << "Maximum time: " << opt.get_maxtime() << "\n";
        std::cout << "Relative x Tolerance: " << opt.get_xtol_rel() << "\n";
        std::cout << "Relative f Tolerance: " << opt.get_ftol_rel() << "\n";
        std::cout << "Absolute f Tolerance: " << opt.get_ftol_abs() << "\n";
    }
    return opt;
}

void TrainMap(std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map, std::shared_ptr<MapObjective<Kokkos::HostSpace>> objective, TrainOptions options) {
    nlopt::opt opt = SetupOptimization(map->numCoeffs, options);
    opt.set_min_objective(objective->GetOptimizationObjective(map), objective.get());
    double error;
    std::vector<double> mapCoeffsStd (map->numCoeffs);
    nlopt::result res = opt.optimize(mapCoeffsStd, error);
    Kokkos::View<double*, Kokkos::HostSpace> mapCoeffs = map->Coeffs();
    std::copy(mapCoeffsStd.begin(), mapCoeffsStd.end(), map->Coeffs().data());
    if(options.verbose){
        std::cout << "Optimization result: " << res << "\n";
        std::cout << "Optimization error: " << error << "\n";
    }
}