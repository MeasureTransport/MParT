#include <map>
#include "MParT/TrainMap.h"

using namespace mpart;

// Vectors for keeping track of NLopt success/failure codes.
const std::vector<std::string> MPART_NLOPT_SUCCESS_CODES {
    "UNDEFINED OPTIMIZATION RESULT",
    "Generic success",
    "stopval reached",
    "xtol reached",
    "xtol reached",
    "maxeval reached",
    "maxtime reached"
};

const std::vector<std::string> MPART_NLOPT_FAILURE_CODES {
    "UNDEFINED OPTIMIZATION RESULT",
    "generic failure",
    "invalid arguments",
    "out of memory",
    "roundoff error limited progress",
    "forced termination"
};

double functor_wrapper(unsigned n, const double *x, double *grad, void *d_) {
    using nloptStdFunction = std::function<double(unsigned,const double*,double*)>;
    nloptStdFunction *obj = reinterpret_cast<nloptStdFunction*>(d_);
    return (*obj)(n, x, grad);
}

nlopt::opt SetupOptimization(unsigned int dim, TrainOptions options) {
    nlopt::opt opt(options.opt_alg.c_str(), dim);

    // Set all the optimization options for nlopt here
    opt.set_stopval(options.opt_stopval);
    opt.set_xtol_rel(options.opt_xtol_rel);
    opt.set_xtol_abs(options.opt_xtol_abs);
    opt.set_ftol_rel(options.opt_ftol_rel);
    opt.set_ftol_abs(options.opt_ftol_abs);
    opt.set_maxeval(options.opt_maxeval);
    opt.set_maxtime(options.opt_maxtime);

    // Print all the optimization options, if verbose
    if(options.verbose){
        std::cout << "Optimization Settings:\n";
        std::cout << "Algorithm: " << opt.get_algorithm_name() << "\n";
        std::cout << "Optimization dimension: " << opt.get_dimension() << "\n";
        std::cout << "Optimization stopval: " << opt.get_stopval() << "\n";
        std::cout << "Max f evaluations: " << opt.get_maxeval() << "\n";
        std::cout << "Maximum time: " << opt.get_maxtime() << "\n";
        std::cout << "Relative x Tolerance: " << opt.get_xtol_rel() << "\n";
        std::cout << "Relative f Tolerance: " << opt.get_ftol_rel() << "\n";
        std::cout << "Absolute f Tolerance: " << opt.get_ftol_abs() << "\n";
    }
    return opt;
}

template<typename MemorySpace>
bool isMapObjectiveStandardGaussianKL(std::shared_ptr<MapObjective<MemorySpace>> objective) {
    std::shared_ptr<KLObjective<MemorySpace>> klObjective = std::dynamic_pointer_cast<KLObjective<MemorySpace>>(objective);
    if(klObjective == nullptr) return false;
    std::shared_ptr<DensityBase<MemorySpace>> density = klObjective->density_;
    std::shared_ptr<GaussianSamplerDensity<MemorySpace>> gaussianDensity = std::dynamic_pointer_cast<GaussianSamplerDensity<MemorySpace>>(density);
    return gaussianDensity != nullptr && gaussianDensity->IsStandardNormal();
}

template<typename MemorySpace>
class NLOptFunctor {
    public:
    NLOptFunctor(std::shared_ptr<MapObjective<MemorySpace>> objective, std::shared_ptr<ConditionalMapBase<MemorySpace>> map): objective_(objective), map_(map) {
        unsigned int numCoeffs = map->numCoeffs;
        coeff_d_ = Kokkos::View<double*, MemorySpace>("coeff", numCoeffs);
        grad_d_ = Kokkos::View<double*, MemorySpace>("grad", numCoeffs);
    }

    double operator()(unsigned n, const double* coeff_ptr, double* grad_ptr) {
        assert(n == coeff_d_.extent(0));
        Kokkos::View<const double*, Kokkos::HostSpace> coeff = ToKokkos<const double, Kokkos::HostSpace>(coeff_ptr, n);
        Kokkos::View<double*, Kokkos::HostSpace> grad = ToKokkos<double, Kokkos::HostSpace>(grad_ptr, n);
        Kokkos::deep_copy(coeff_d_, coeff);
        double error;
        if(grad_ptr != nullptr) {
            error = (*objective_)(n, coeff_d_.data(), grad_d_.data(), map_);
            Kokkos::deep_copy(grad, grad_d_);
            Kokkos::deep_copy(grad_d_, 0);
        } else {
            error = (*objective_)(n, coeff_d_.data(), nullptr, map_);
        }
        return error;
    }

    private:
    Kokkos::View<double*, MemorySpace> coeff_d_;
    Kokkos::View<double*, MemorySpace> grad_d_;
    std::shared_ptr<MapObjective<MemorySpace>> objective_;
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;
};

template<typename MemorySpace>
std::function<double(unsigned, const double*, double*)> CreateNLOptObjective(std::shared_ptr<MapObjective<MemorySpace>> objective, std::shared_ptr<ConditionalMapBase<MemorySpace>> map) {
    if constexpr(std::is_same_v<MemorySpace, Kokkos::HostSpace>) {
        return std::bind(&MapObjective<MemorySpace>::operator(), objective, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, map);
    } else {
        NLOptFunctor<MemorySpace> functor(objective, map);
        return functor;
    }
}

template<typename MemorySpace>
double TrainComponent_StandardNormalDensity(std::shared_ptr<ConditionalMapBase<MemorySpace>> component, std::shared_ptr<FastGaussianReverseKLObjective<MemorySpace>> objective, TrainOptions options) {
    if(component->outputDim != 1) {
        throw std::runtime_error("TrainComponent_StandardNormalDensity: Component must be scalar-valued. Has output dimension " + std::to_string(component->outputDim) + ".");
    }
    std::shared_ptr<MapObjective<MemorySpace>> mapObjective = std::dynamic_pointer_cast<MapObjective<MemorySpace>>(objective);
    assert(mapObjective != nullptr);
    std::function<double(unsigned, const double*, double*)> functor = CreateNLOptObjective(mapObjective, component);

    // Get the initial guess at the coefficients
    std::vector<double> compCoeffsStd = KokkosToStd(component->Coeffs());

    // Optimize the map coefficients using NLopt
    nlopt::opt opt = SetupOptimization(component->numCoeffs, options);
    opt.set_min_objective(functor_wrapper, reinterpret_cast<void*>(&functor));

    double error = 0.;
    
    nlopt::result res = opt.optimize(compCoeffsStd, error);
    Kokkos::View<const double*, Kokkos::HostSpace> compCoeffsView = VecToKokkos<double,Kokkos::HostSpace>(compCoeffsStd);
    component->SetCoeffs(compCoeffsView);

    if(options.verbose) {
        if(res < 0) {
            std::cerr << "WARNING: Optimization failed: " << MPART_NLOPT_FAILURE_CODES[-res] << std::endl;
        } else {
            std::cout << "Optimization result: " << MPART_NLOPT_SUCCESS_CODES[res] << std::endl;
        }
        std::cout << "Optimization error: " << error << "\n"
                  << "Optimization evaluations: " << opt.get_numevals() << std::endl;
    }
    return error;
}

template<typename MemorySpace>
double TrainMap_Triangular_StandardNormalDensity(std::shared_ptr<TriangularMap<MemorySpace>> map, std::shared_ptr<KLObjective<MemorySpace>> objective, TrainOptions options) {
    if(options.verbose) {
        std::cout << "Detected standard normal reference density with Triangular Map" << std::endl;
    }
    double total_error = 0.;
    StridedMatrix<const double, MemorySpace> trainSamples = objective->GetTrain();
    // Assume each component is a scalar-valued function
    for(int i = 0; i < map->NumComponents(); i++) {
        std::shared_ptr<ConditionalMapBase<MemorySpace>> component = map->GetComponent(i);
        // Create a new objective for each component by slicing the KL objective
        StridedMatrix<const double, MemorySpace> trainSlice = Kokkos::subview(trainSamples, Kokkos::make_pair(0u, component->inputDim), Kokkos::ALL());
        std::shared_ptr<FastGaussianReverseKLObjective<MemorySpace>> componentObjective = std::make_shared<FastGaussianReverseKLObjective<MemorySpace>>(trainSlice, component->numCoeffs);
        if(options.verbose) std::cout << "Training component " << i << "..." << std::endl;
        total_error += TrainComponent_StandardNormalDensity(component, componentObjective, options);
        if(options.verbose) {
            std::cout << "Component " << i << " trained.\n" << std::endl;
        }
    }
    return total_error;
}

template<typename MemorySpace>
double mpart::TrainMap(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<MapObjective<MemorySpace>> objective, TrainOptions options) {
    if(map->Coeffs().extent(0) == 0) {
        if(options.verbose) {
            std::cout << "TrainMap: Initializing map coeffs to 1." << std::endl;
        }
        Kokkos::View<double*, Kokkos::HostSpace> coeffs ("Default coeffs", map->numCoeffs);
        Kokkos::deep_copy(coeffs, 1.);
	    Kokkos::View<const double*, Kokkos::HostSpace> constCoeffs = coeffs;
        map->SetCoeffs(constCoeffs);
    }
    std::shared_ptr<TriangularMap<MemorySpace>> tri_map = std::dynamic_pointer_cast<TriangularMap<MemorySpace>>(map);
    bool isMapTriangular = tri_map != nullptr;
    if(isMapObjectiveStandardGaussianKL(objective) && isMapTriangular) {
        std::shared_ptr<KLObjective<MemorySpace>> klObjective = std::dynamic_pointer_cast<KLObjective<MemorySpace>>(objective);
        return TrainMap_Triangular_StandardNormalDensity(tri_map, klObjective, options);
    }

    nlopt::opt opt = SetupOptimization(map->numCoeffs, options);

    // Since objective is (rightfully) separate from the map, we use std::bind to create a functor
    // from objective::operator() that keeps the map argument held.
    std::function<double(unsigned, const double*, double*)> functor = CreateNLOptObjective(objective, map);
    opt.set_min_objective(functor_wrapper, reinterpret_cast<void*>(&functor));

    // Get the initial guess at the coefficients
    std::vector<double> mapCoeffsStd = KokkosToStd(map->Coeffs());

    // Optimize the map coefficients using NLopt
    double error;
    nlopt::result res = opt.optimize(mapCoeffsStd, error);

    // Set the coefficients using SetCoeffs
    Kokkos::View<const double*, Kokkos::HostSpace> mapCoeffsView = VecToKokkos<double,Kokkos::HostSpace>(mapCoeffsStd);
    map->SetCoeffs(mapCoeffsView);

    // Print a warning if something goes wrong with NLOpt
    if(res < 0) {
        std::cerr << "WARNING: Optimization failed: " << MPART_NLOPT_FAILURE_CODES[-res] << std::endl;
    }

    // Print results of optimization if verbose
    if(options.verbose){
        if(res >= 0){
            std::cout << "Optimization result: " << MPART_NLOPT_SUCCESS_CODES[res] << "\n";
        }
        std::cout << "Optimization error: " << error << "\n";
        std::cout << "Optimization evaluations: " << opt.get_numevals() << std::endl;
    }
    return error;
}

template double mpart::TrainMap<Kokkos::HostSpace>(std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map, std::shared_ptr<MapObjective<Kokkos::HostSpace>> objective, TrainOptions options);
#if defined(MPART_ENABLE_GPU)
template double mpart::TrainMap<DeviceSpace>(std::shared_ptr<ConditionalMapBase<DeviceSpace>> map, std::shared_ptr<MapObjective<DeviceSpace>> objective, TrainOptions options);
#endif