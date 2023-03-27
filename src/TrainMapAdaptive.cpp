#include "MParT/TrainMapAdaptive.h"
#include <fstream>

using namespace mpart;

void findMaxGrad(StridedVector<double, Kokkos::HostSpace> const &gradCoeff, std::vector<std::vector<unsigned int>> const &multis_rm,
    std::vector<MultiIndexSet> const &msets, unsigned int &maxIdx, unsigned int &maxIdxBlock) {
    unsigned int outputDim = multis_rm.size();
    double maxVal = 0.;
    unsigned int blockStart = 0;
    for(int output = 0; output < outputDim; output++) {
        for(int i = 0; i < multis_rm[output].size(); i++) {
            unsigned int idx = multis_rm[output][i];
            double gradVal = std::abs(gradCoeff(blockStart + idx));
            if(gradVal > maxVal) {
                maxVal = gradVal;
                maxIdx = idx;
                maxIdxBlock = output;
            }
        }
        blockStart += msets[output].Size();
    }
}

void maxDegreeRMFilter(std::vector<MultiIndexSet> const &msets, MultiIndex const &maxDegrees,
    std::vector<std::vector<unsigned int>> &multis_rm) {
    unsigned int outputDim = msets.size();
    for(unsigned int j = 0; j < outputDim; j++) {
        std::vector<unsigned int>& rm_j = multis_rm[j];
        const MultiIndexSet& mset_j = msets[j];
        std::vector<bool> bounded = mset_j.FilterBounded(maxDegrees);
        for(std::vector<unsigned int>::iterator it = rm_j.begin(); it != rm_j.end();) {
            if(bounded[*it]) {
                it = rm_j.erase(it);
            } else {
                it++;
            }
        }
    }
}


template<>
std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::TrainMapAdaptive(std::vector<MultiIndexSet> &mset0,
        KLObjective<Kokkos::HostSpace> &objective,
        ATMOptions options) {

    // Dimensions
    unsigned int inputDim = objective.Dim();
    unsigned int outputDim = mset0.size();

    std::vector<unsigned int> mset_sizes (outputDim);
    unsigned int currSz = 0;
    unsigned int currPatience = 0;
    std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> mapBlocks (outputDim);
    std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> mapBlocksTmp (outputDim);

    // Setup Multiindex info for training
    std::vector<std::vector<unsigned int>> multis_rm (outputDim);
    std::vector<MultiIndexSet> mset_tmp {};
    std::vector<MultiIndexSet> mset_best {};

    // Ensure the given vector of multiindexsets is valid
    unsigned int currMsetDim = mset0[0].Length();
    for(unsigned int j = 0; j < outputDim; j++) {

        MultiIndexSet mset_j = mset0[j];
        mset_sizes[j] = mset_j.Size();

        // Check the length is okay
        if(mset_j.Length() != currMsetDim) {
            std::stringstream ss;
            ss << "TrainMapAdaptive: Initial MultiIndexSet " << j << " is invalid.\n";
            ss << "Expected Length " << currMsetDim << ", got " << mset_j.Length() << ".";
            throw std::invalid_argument(ss.str());
        }

        // Create the jth component
        mapBlocks[j] = MapFactory::CreateComponent(mset_j.Fix(true), options);

        // Initialize the temporary mset vectors
        MultiIndexSet mset_tmp_j = MultiIndexSet::CreateTotalOrder(inputDim-outputDim+j+1,0);
        mset_tmp.push_back(mset_tmp_j);
        mset_best.push_back(mset_tmp_j);

        // Index global constants
        currMsetDim++;
        currSz += mset_sizes[j];
    }
    currMsetDim--; // Adjust for the extra ++

    // Check that the multiindex sets match with the objective
    if(currMsetDim != inputDim) {
        std::stringstream ss;
            ss << "TrainMapAdaptive: Initial MultiIndexSets must match Objective dimension.\n";
            ss << "Expected last MultiIndexSet to have Length " << inputDim << ", got " << currMsetDim << ".";
            throw std::invalid_argument(ss.str());
    }

    // Ensure that the dimensional maximum degree vector is valid
    bool hasMaxDegrees = true;
    if(options.maxDegrees.Length() == 0) {
        hasMaxDegrees = false;
    } else if(options.maxDegrees.Length() != currMsetDim){
        std::stringstream ss;
        ss << "TrainMapAdaptive: invalid options.maxDegrees for this vector of MultiIndexSet objects\n";
        ss << "Expected length either zero or " << currMsetDim << "\n";
        throw std::invalid_argument(ss.str());
    }

    // Create initial map
    std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocks);
    Kokkos::View<double*, Kokkos::HostSpace> mapCoeffs ("mapCoeffs", map->numCoeffs);
    for(int i = 0; i < mapCoeffs.extent(0); i++) {
        mapCoeffs(i) = 0.;
    }
    map->WrapCoeffs(mapCoeffs);

    if(options.verbose) {
        std::cout << "Initial map:" << std::endl;
    }
    TrainMap(map, objective, options);
    double bestError = objective.TestError(map);

    if(options.verbose) {
        std::cout << "Initial map test error: " << bestError << std::endl;
    }

    // Perform ATM
    while(currSz < options.maxSize && currPatience < options.maxPatience) {
        if(options.verbose) {
            std::cout << "\n\nSize " << currSz << ":" << std::endl;
        }

        for(int i = 0; i < outputDim; i++) {
            // Expand the current map
            mset_tmp[i] = mset0[i];
            multis_rm[i] = mset_tmp[i].Expand();

            // Create a component with the expanded multiindexset
            std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> comp_i = MapFactory::CreateComponent(mset_tmp[i].Fix(), options);
            Kokkos::View<double*,Kokkos::HostSpace> coeffsFrontier ("Frontier coeffs", comp_i->numCoeffs);
            int oldIdx = 0;
            int rmIdx = 0;
            for(int j = 0; j < comp_i->numCoeffs; j++) {
                if(j == multis_rm[i][rmIdx]){ // Fill new coeff with zero
                    coeffsFrontier(j) = 0.;
                    rmIdx++;
                }
                else{ // Copy the old coefficients from the current map to the new component
                    coeffsFrontier(j) = mapBlocks[i]->Coeffs()(oldIdx);
                    oldIdx++;
                }
            }
            comp_i->WrapCoeffs(coeffsFrontier);
            // Set the new component
            mapBlocksTmp[i] = comp_i;
        }
        // Create a temporary map
        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mapTmp = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocksTmp, true);
        // Calculate the gradient of the map with expanded margins
        StridedVector<double, Kokkos::HostSpace> gradCoeff = objective.TrainCoeffGrad(mapTmp);
        int coeffIdx = 0;
        if(options.verbose > 1) {
            for(int output=0; output<outputDim; output++){
                int rmIdx = 0;
                for(int i = 0; i < mset_tmp[output].Size(); i++) {
                    std::cout << "gradCoeff(" << coeffIdx << ")=" << gradCoeff(coeffIdx) << ", midx=[" << mset_tmp[output][i] << "]";
                    if(i == multis_rm[output][rmIdx]) {
                        std::cout << " rm!";
                        rmIdx++;
                    }
                    std::cout << std::endl;
                    coeffIdx++;
                }
            }
        }

        // Filter out the reduced margin elements that meet or exceed the maximum
        if(hasMaxDegrees) {
            maxDegreeRMFilter(mset_tmp, options.maxDegrees, multis_rm);
        }

        // Find the largest gradient value, calculate which output and multiindex it corresponds to
        unsigned int maxIdx, maxIdxBlock;
        findMaxGrad(gradCoeff, multis_rm, mset_tmp, maxIdx, maxIdxBlock);

        // Add the multiindex with largest gradient to the map
        MultiIndex addedMulti = mset_tmp[maxIdxBlock][maxIdx];
        mset0[maxIdxBlock] += addedMulti;
        if(options.verbose) {
            std::cout << "Added multi = [" << addedMulti.String() << "]" <<std::endl;
        }
        currSz++;
        if(mset0[maxIdxBlock].Size() != mset_sizes[maxIdxBlock]+1) {
            std::cerr << addedMulti << "\n";
            std::stringstream ss_err;
            ss_err << "TrainMapAdaptive: Unexpected sizes of MultiIndexSets.\n";
            ss_err << "mset0["<<maxIdxBlock<<"].Size() = " << mset0[maxIdxBlock].Size() << ", mset_sizes["<<maxIdxBlock<<"] = " << mset_sizes[maxIdxBlock]+1;
            throw std::runtime_error(ss_err.str());
        }

        // Create a new map with the larger MultiIndexSet in component maxIdxBlock
        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> newComp = MapFactory::CreateComponent(mset0[maxIdxBlock].Fix(true), options);
        Kokkos::View<double*, Kokkos::HostSpace> oldCoeffs = mapBlocks[maxIdxBlock]->Coeffs();
        Kokkos::View<double*, Kokkos::HostSpace> newCoeffs ("New component coefficients", newComp->numCoeffs);
        std::copy(oldCoeffs.data(), oldCoeffs.data() + mset_sizes[maxIdxBlock], newCoeffs.data());
        newCoeffs(mset_sizes[maxIdxBlock]) = 0.;
        mapBlocks[maxIdxBlock] = newComp;
        mset_sizes[maxIdxBlock]++;
        map = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocks, true);

        // Train a map with the new MultiIndex
        double train_error = TrainMap(map, objective, options);
        // Get the testing error and assess the best map
        double test_error = objective.TestError(map);

        // Finish this step
        currPatience++;
        if(test_error < bestError) {
            bestError = test_error;
            for(int i = 0; i < outputDim; i++) {
                mset_best[i] = mset0[i];
            }
            // Reset the patience if we find a new best map
            currPatience = 0;
        }

        // Print the current iteration results if verbose
        if(options.verbose) {
            std::cout << "Size " << currSz-1 << " complete. Train error: " << train_error << " Test error: " << test_error << std::endl;
        }
    }

    // Create the "best" map components based on the best MultiIndexSets
    for(int i = 0; i < outputDim; i++) {
        mapBlocks[i] = MapFactory::CreateComponent(mset_best[i].Fix(true), options);
        mset0[i] = mset_best[i];
    }

    // Create the "best" map and optimize it
    map = std::make_shared<TriangularMap<Kokkos::HostSpace>>(mapBlocks, true);
    if(options.verbose) {
        std::cout << "\nFinal map has " << map->numCoeffs << " terms.\n";
    }
    // Train a map with the new MultiIndex
    double train_error = TrainMap(map, objective, options);
    // Get the testing error and assess the best map
    double test_error = objective.TestError(map);

    if(options.verbose) {
        std::cout << "\nFinal training error: " << train_error << ", final testing error: " << test_error;
        if(options.verbose > 1) {
            std::cout << "\nVisualization of MultiIndexSets--\n";
            for(int i = 0; i < outputDim; i++) {
                std::cout << "mset " << i << ":\n";
                mset0[i].Visualize();
                std::cout << "\n";
            }
        }
        std::cout << std::endl;
    }
    return map;
}
