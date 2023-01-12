#include "MParT/AdaptiveTransportMap.h"

using namespace mpart;

template<>
double ATMObjective<Kokkos::HostSpace>::operator()(unsigned int n, const double* coeffs, double* grad) {
    const unsigned int N_samps = x_.extent(1);
    StridedVector<const double, Kokkos::HostSpace> coeffView = ToConstKokkos<double, Kokkos::HostSpace>(coeffs, n);

    // Rest of function is agnostic of MemorySpace, so keep it generic
    using MemorySpace = Kokkos::HostSpace;
    std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density = std::make_shared<GaussianSamplerDensity<MemorySpace>>(x_.extent(0));
    map_->SetCoeffs(coeffView);
    PullbackDensity<MemorySpace> pullback {map_, density};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(x_);
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(x_);
    double sumDensity = 0.;
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);
    ReduceColumn rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, grad);
    return sumDensity/N_samps;
}

template<>
void ATMObjective<Kokkos::HostSpace>::Gradient(unsigned int n, const double* coeffs, double* grad) {
    const unsigned int N_samps = x_.extent(1);
    StridedVector<const double, Kokkos::HostSpace> coeffView = ToConstKokkos<double, Kokkos::HostSpace>(coeffs, n);

    // Rest of function is agnostic of MemorySpace, so keep it generic
    using MemorySpace = Kokkos::HostSpace;
    std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density = std::make_shared<GaussianSamplerDensity<MemorySpace>>(x_.extent(0));
    map_->SetCoeffs(coeffView);
    PullbackDensity<MemorySpace> pullback {map_, density};
    StridedMatrix<double, MemorySpace> densityGradX = pullback.LogDensityCoeffGrad(x_);
    ReduceColumn rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, grad);
}

template<>
double ATMObjective<Kokkos::HostSpace>::TestError(StridedVector<const double, Kokkos::HostSpace> coeffView) {
    const unsigned int N_samps = x_.extent(1);
    // Rest of function is agnostic of MemorySpace, so keep it generic
    using MemorySpace = Kokkos::HostSpace;
    std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density = std::make_shared<GaussianSamplerDensity<MemorySpace>>(x_test_.extent(0));
    map_->SetCoeffs(coeffView);
    PullbackDensity<MemorySpace> pullback {map_, density};
    StridedVector<double, MemorySpace> densityX = pullback.LogDensity(x_test_);
    double sumDensity = 0.;
    Kokkos::parallel_reduce ("Sum Negative Log Likelihood, Test", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i);
    }, sumDensity);
    return sumDensity/N_samps;
}

nlopt::opt SetupOptimization(unsigned int dim, ATMOptions options) {
    nlopt::opt opt(options.opt_alg.c_str(), dim);
    // TODO: Set all the optimization options here
    if(options.opt_xtol_rel > 0)
        opt.set_xtol_rel(options.opt_xtol_rel);

    // TODO: Print out all of the optimization settings here if verbose
    if(options.verbose){ }
    return opt;
}

// Used for calculating the best location
bool largestAbs(double a, double b) {
    return std::abs(a) < std::abs(b);
}

template<>
std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::AdaptiveTransportMap(std::vector<MultiIndexSet> &mset0,
                StridedMatrix<double, Kokkos::HostSpace> train_x, StridedMatrix<double, Kokkos::HostSpace> test_x,
                ATMOptions options) {
    // Dimensions
    unsigned int inputDim = train_x.extent(0);
    unsigned int outputDim = mset0.size();

    // Setup initial map
    std::vector<unsigned int> mset_sizes (outputDim);
    unsigned int currSz = 0;
    unsigned int currPatience = 0;
    std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> mapBlocks (outputDim);
    std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> mapBlocksTmp (outputDim);
    for(int i = 0; i < outputDim; i++) {
        mset_sizes[i] = mset0[i].Size();
        mapBlocks[i] = MapFactory::CreateComponent(mset0[i].Fix(true), options);
        currSz += mset_sizes[i];
    }
    // Create initial map
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocks);
    Kokkos::View<double*, Kokkos::HostSpace> mapCoeffs ("mapCoeffs", map->numCoeffs);
    map->WrapCoeffs(mapCoeffs);
    // Create a tmp map
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mapTmp;
    // Initialize the optimization objective
    ATMObjective<Kokkos::HostSpace> objective(train_x, test_x, map, options);

    // Setup optimization
    nlopt::opt opt = SetupOptimization(map->numCoeffs, options);
    opt.set_min_objective(objective);

    // Train initial map
    double bestError;
    std::vector<double> mapCoeffsStd = KokkosToStd(map->Coeffs());
    opt.optimize(mapCoeffsStd, bestError);

    if(options.verbose)
        std::cout << "Initial map trained with error: " << bestError << std::endl;

    // Setup Multiindex info for training
    std::vector<std::vector<unsigned int>> multis_rm (outputDim);
    MultiIndexSet msetDefault = MultiIndexSet::CreateTotalOrder(inputDim, 0);
    std::vector<MultiIndexSet> mset_tmp {};
    std::vector<MultiIndexSet> mset_best {};
    for(int i = 1; i <= outputDim; i++) {
        MultiIndexSet mset_i = MultiIndexSet::CreateTotalOrder(inputDim-outputDim+i,0);
        mset_tmp.push_back(mset_i);
        mset_best.push_back(mset_i);
    }

    while(currSz < options.maxSize && currPatience < options.maxPatience) {
        for(int i = 0; i < outputDim; i++) {
            // Expand the current map
            mset_tmp[i] = mset0[i];
            multis_rm[i] = mset_tmp[i].Expand();
            // Create a component with the expanded multiindexset
            std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> comp_i = MapFactory::CreateComponent(mset_tmp[i].Fix(), options);
            // Copy the old coefficients from the current map to the new component
            double* map_coeff_i = mapBlocks[i]->Coeffs().data();
            double* comp_coeff_i = comp_i->Coeffs().data();
            std::copy(map_coeff_i, map_coeff_i + mset_sizes[i], comp_coeff_i);
            // Fill the rest of the coefficients with zeros
            std::fill(comp_coeff_i + mset_sizes[i], comp_coeff_i + comp_i->Coeffs().extent(0), 0.);
            // Set the new component
            mapBlocksTmp[i] = comp_i;
        }
        // Create a temporary map
        mapTmp = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocksTmp, true);
        // Calculate the gradient of the map with expanded margins
        objective.SetMap(mapTmp);
        std::vector<double> gradCoeff (mapTmp->Coeffs().extent(0));
        objective.Gradient(mapTmp->numCoeffs, mapTmp->Coeffs().data(), gradCoeff.data());

        // Find the largest gradient value, calculate which output and multiindex it corresponds to
        int maxIdx = std::distance(gradCoeff.begin(), std::max_element(gradCoeff.begin(), gradCoeff.end(), largestAbs));
        int maxIdxBlock = 0;
        int maxIdxBlockOffset = 0;
        for(int i = 0; i < outputDim; i++) {
            if(maxIdx < mset_sizes[i]) {
                maxIdxBlock = i;
                break;
            }
            maxIdxBlockOffset += mset_sizes[i];
            maxIdx -= mset_sizes[i];
        }
        // Add the multiindex with largest gradient to the map
        unsigned int idxNew = multis_rm[maxIdxBlock][maxIdx];
        mset0[maxIdxBlock] += mset_tmp[maxIdxBlock][idxNew];

        // Create a new map with the larger MultiIndexSet in component maxIdxBlock
        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> newComp = MapFactory::CreateComponent(mset0[maxIdxBlock].Fix(true), options);
        Kokkos::View<double*, Kokkos::HostSpace> oldCoeffs = mapBlocks[maxIdxBlock]->Coeffs();
        Kokkos::View<double*, Kokkos::HostSpace> newCoeffs = newComp->Coeffs();
        std::copy(oldCoeffs.data(), oldCoeffs.data() + mset_sizes[maxIdxBlock], newCoeffs.data());
        std::fill(newCoeffs.data() + mset_sizes[maxIdxBlock], newCoeffs.data() + newCoeffs.extent(0), 0.);
        mapBlocks[maxIdxBlock] = newComp;
        map = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocks, true);

        // Train a map with the new MultiIndex
        objective.SetMap(map);
        double train_error;
        mapCoeffsStd = KokkosToStd(map->Coeffs());
        opt = SetupOptimization(map->numCoeffs, options);
        opt.set_min_objective(objective);
        opt.optimize(mapCoeffsStd, train_error);
        // Get the testing error and assess the best map
        double test_error = objective.TestError(map->Coeffs());
        if(test_error < bestError) {
            bestError = test_error;
            for(int i = 0; i < outputDim; i++) {
                mset_best[i] = mset0[i];
            }
            // Reset the patience if we find a new best map
            currPatience = 0;
        } else {
            currPatience++;
        }

        // Print the current iteration results if verbose
        if(options.verbose) {
            std::cout << "Iteration " << currSz << " complete. Train error: " << train_error << " Test error: " << test_error << std::endl;
        }
        currSz++;
    }
    // Create the "best" map components based on the best MultiIndexSets
    for(int i = 0; i < outputDim; i++) {
        mapBlocks[i] = MapFactory::CreateComponent(mset_best[i].Fix(true), options);
        mset0[i] = mset_best[i];
        // Initialize coeffs as zero
        Kokkos::View<double*, Kokkos::HostSpace> coeffs_i = mapBlocks[i]->Coeffs();
        std::fill(coeffs_i.data(), coeffs_i.data() + coeffs_i.extent(0), 0.);
    }

    // Create the "best" map and optimize it
    map = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocks, true);
    objective.SetMap(map);
    double train_error;
    mapCoeffsStd = KokkosToStd(map->Coeffs());
    opt.optimize(mapCoeffsStd, train_error);

    // Get the testing error, printing information if verbose
    double test_error = objective.TestError(map->Coeffs());
    if(options.verbose) {
        std::cout << "Final map with " << map->numCoeffs << " terms trains with error: " << train_error << std::endl;
    }
    return map;
}