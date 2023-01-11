#include "MParT/AdaptiveTransportMap.h"

using namespace mpart;

template<>
ATMObjective<Kokkos::HostSpace>::ATMObjective(StridedMatrix<double, Kokkos::HostSpace> x,
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map, ATMOptions options = ATMOptions()):
    x_(x), map_(map), options_(options) {
    if(options.referenceType != ReferenceTypes::StandardGaussian) {
        throw std::invalid_argument("ATMObjective<Kokkos::HostSpace>::ATMObjective: Currently only accepts Gaussian reference")
    }
}

template<>
double ATMObjective<Kokkos::HostSpace>::operator()(const std::vector<double> &coeffs, std::vector<double> &grad) {
    const unsigned int N_samps = x_.extent(1);
    StridedVector<double, Kokkos::HostSpace> coeffView = VecToKokkos(coeffs);
    StridedVector<double, Kokkos::HostSpace> gradView = VecToKokkos(grad);

    std::shared_ptr<GaussianSamplerDensity<Kokkos::HostSpace>> reference = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(x_.extent(0));
    map_->WrapCoeffs(coeffView);
    PullbackDensity<MemorySpace> pullback {map_, reference};
    StridedVector<double, Kokkos::HostSpace> densityX = pullback.LogDensity(x_);
    StridedMatrix<double, Kokkos::HostSpace> densityGradX = pullback.LogDensityCoeffGrad(x_);
    double sumDensity = 0.;
    Kokkos::parallel_reduce ("Average Negative Log Likelihood", N_samps, KOKKOS_LAMBDA (const int i, double &sum) {
        sum -= densityX(i)/N_samps;
    }, sumDensity);
    ReduceColumn rc(densityGradX, -1.0/((double) N_samps));
    Kokkos::parallel_reduce(N_samps, rc, gradView.data());
}

nlopt::opt SetupOptimization(unsigned int dim, ATMOptions options) {
    nlopt::opt opt(options::opt_alg, dim);
    opt.set_min_objective(objective, nullptr);
    // TODO: Set all the optimization options here
    if(options::opt_xtol_rel > 0)
        opt.set_xtol_rel(options::opt_xtol_rel);

    // TODO: Print out all of the optimization settings here if verbose
    if(options::verbose){ }
}

bool largestAbs(double a, double b) {
    return std::abs(a) < std::abs(b);
}

template<>
std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> AdaptiveTransportMap(std::vector<MultiIndexSet> &mset0,
                StridedMatrix<double, MemorySpace> train_x, StridedMatrix<double, MemorySpace> test_x,
                ATMOptions options) {
    unsigned int inputDim = train_x.extent(0);
    unsigned int outputDim = mset0.size();
    std::vector<unsigned int> mset_sizes (outputDim);
    unsigned int currSz = 0;
    std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> mapBlocks (outputDim);
    std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> mapBlocksTmp (outputDim);
    for(int i = 0; i < outputDim; i++) {
        mset_sizes[i] = mset0[i].Size();
        mapBlocks[i] = CreateComponent(mset0[i].Fix(true), options);
        currSz += mset_sizes[i];
    }
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocks);
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mapTmp;
    ATMObjective<MemorySpace> objective(train_x, test_x, map, options);
    std::vector<double> coeffMap = KokkosToVec(map->Coeffs());

    // Setup optimization
    nlopt::opt opt = SetupOptimization(coeffMap.size(), options);

    double bestError;
    opt.optimize(coeffMap, bestError);

    if(options::verbose)
        std::cout << "Initial map trained with error: " << initMapError << std::endl;

    std::vector<std::vector<unsigned int>> multis_rm (outputDim);
    std::vector<MultiIndexSet> mset_temp(outputDim);
    std::vector<MultiIndexSet> mset_best(outputDim);

    while(currSz < options.maxSize && currPatience < options.maxPatience) {
        for(int i = 0; i < outputDim; i++) {
            mset_temp[i] = mset[i];
            multis_rm[i] = mset_temp[i].Expand();
            std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> comp_i = CreateComponent(mset_temp[i].Fix(true), options);
            std::copy(mapBlocks[i]->Coeffs().data(), mapBlocks[i]->Coeffs().data() + mset_sizes[i], comp_i->Coeffs().data());
            std::fill(comp_i->Coeffs().data() + mset_sizes[i], comp_i->Coeffs().data() + comp_i->Coeffs().extent(0), 0.);
            mapBlocksTmp[i] = comp_i;
        }

        mapTmp = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocksTmp, true);
        objective->SetMap(mapTmp);
        std::vector<double> gradCoeff (mapTmp->Coeffs().extent(0));
        double error = objective(mapTmp->Coeffs(), gradCoeff);
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
        unsigned int idxNew = multis_rm[maxIdxBlock][maxIdx];
        MultiIndex multiNew = mset_temp[maxIdxBlock].at(idxNew);
        mset[maxIdxBlock] += multiNew;
        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> newComp = CreateComponent(mset[maxIdxBlock].Fix(true), options);
        std::copy(mapBlocks[maxIdxBlock]->Coeffs().data(), mapBlocks[maxIdxBlock]->Coeffs().data() + mset_sizes[maxIdxBlock], newComp->Coeffs().data());
        std::fill(newComp->Coeffs().data() + mset_sizes[maxIdxBlock], newComp->Coeffs().data() + newComp->Coeffs().extent(0), 0.);
        mapBlocks[maxIdxBlock] = newComp;
        map = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocks, true);
        objective->SetMap(map);
        coeffMap = KokkosToVec(map->Coeffs());
        double train_error;
        opt.optimize(coeffMap, train_error);
        double test_error = objective->TestError();
        if(test_error < bestError) {
            bestError = test_error;
            for(int i = 0; i < outputDim; i++) {
                mset_best[i] = mset[i];
            }
            currPatience = 0;
        } else {
            currPatience++;
        }

        if(options::verbose) {
            std::cout << "Iteration " << currSz << " complete. Train error: " << train_error << " Test error: " << test_error << std::endl;
        }
    }
    for(int i = 0; i < outputDim; i++) {
        mapBlocks[i] = CreateComponent(mset_best[i].Fix(true), options);
        mset0[i] = mset_best[i];
        std::fill(mapBlocks[i]->Coeffs().data(), mapBlocks[i]->Coeffs().data() + mset_best[i].Size(), 0.);
    }
    map = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocks, true);
    objective->SetMap(map);
    coeffMap = KokkosToVec(map->Coeffs());
    double train_error;
    opt.optimize(coeffMap, train_error);
    double test_error = objective->TestError();
    if(options::verbose) {
        std::cout << "Final map with " << coeffMap.size() << " terms trains with error: " << train_error << std::endl;
    }
    return map;
}