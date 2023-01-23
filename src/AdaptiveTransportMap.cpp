#include "MParT/AdaptiveTransportMap.h"
#include <fstream>

using namespace mpart;

// Used for calculating the best location
bool largestAbs(double a, double b) {
    return std::abs(a) < std::abs(b);
}

void SaveMatrix(std::string fname, StridedMatrix<double, Kokkos::HostSpace> mat, std::string path="/home/dannys4/misc/mpart_atm/") {
    std::fstream file {path + fname, std::ios::out};
    for(int i = 0; i < mat.extent(0); i++) {
        for(int j = 0; j < mat.extent(1); j++) {
            file << mat(i,j);
            if(j < mat.extent(1)-1) file << ",";
        }
        file <<"\n";
    }
}

template<>
std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::AdaptiveTransportMap(std::vector<MultiIndexSet> &mset0,
        KLObjective<Kokkos::HostSpace> &objective,
        ATMOptions options) {
    // Dimensions
    unsigned int inputDim = objective.Dim();
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

    // Setup Multiindex info for training
    std::vector<std::vector<unsigned int>> multis_rm (outputDim);
    std::vector<MultiIndexSet> mset_tmp {};
    std::vector<MultiIndexSet> mset_best {};
    for(int i = 1; i <= outputDim; i++) {
        MultiIndexSet mset_i = MultiIndexSet::CreateTotalOrder(inputDim-outputDim+i,0);
        mset_tmp.push_back(mset_i);
        mset_best.push_back(mset_i);
    }

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
        for(int output=0; output<outputDim; output++){
            int rmIdx = 0;
            for(int i = 0; i < mset_tmp[output].Size(); i++) {
                std::cerr << "gradCoeff(" << coeffIdx << ")=" << gradCoeff(coeffIdx) << ", midx=[" << mset_tmp[output][i] << "]";
                if(i == multis_rm[output][rmIdx]) {
                    std::cerr << " rm!";
                    rmIdx++;
                }
                std::cerr << std::endl;
                coeffIdx++;
            }
        }


        // Find the largest gradient value, calculate which output and multiindex it corresponds to
        unsigned int maxIdx = 0;
        unsigned int maxIdxBlock = 0;
        double maxVal = 0.;
        double signMaxVal = 0;
        coeffIdx = 0;
        for(int output = 0; output < outputDim; output++) {
            for(int i = 0; i < multis_rm[output].size(); i++) {
                unsigned int idx = multis_rm[output][i];
                double gradVal = gradCoeff(coeffIdx + idx);
                if(std::abs(gradVal) > maxVal) {
                    signMaxVal = gradVal;
                    maxVal = std::abs(gradVal);
                    maxIdx = idx;
                    maxIdxBlock = output;
                }
            }
            coeffIdx += mset_tmp[output].Size();
        }
        // Add the multiindex with largest gradient to the map
        std::cerr << "Before: mset0[" << maxIdxBlock << "].Size() = " << mset0[maxIdxBlock].Size() << ", mset_sizes[" << maxIdxBlock << "] = " << mset_sizes[maxIdxBlock] << "\n";
        mset0[maxIdxBlock] += mset_tmp[maxIdxBlock][maxIdx];
        std::cerr << "After: mset0[" << maxIdxBlock << "].Size() = " << mset0[maxIdxBlock].Size() << std::endl;
        MultiIndex addedMulti = mset_tmp[maxIdxBlock][maxIdx];
        std::cerr << "Added Gradient: " << signMaxVal << ", added multi = [" << addedMulti.String() << "]" <<std::endl;
        currSz++;
        if(mset0[maxIdxBlock].Size() != mset_sizes[maxIdxBlock]+1) {
            std::cerr << mset_tmp[maxIdxBlock][maxIdx] << "\n";
            std::stringstream ss_err;
            ss_err << "maxIdxBlock = " << maxIdxBlock << ", mset0[maxIdxBlock].Size() = " << mset0[maxIdxBlock].Size() << ", mset_sizes[maxIdxBlock] = " << mset_sizes[maxIdxBlock];
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

        // DEBUG
        auto train = objective.GetTrain();
        auto evalSamples = map->Evaluate(train);
        SaveMatrix("sz" + std::to_string(currSz) + "map.csv", evalSamples);
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
        std::cout << "Training error: " << train_error << ", Testing error: " << test_error << "\n";
        std::cout << "Visualization of MultiIndexSets--\n";
        for(int i = 0; i < outputDim; i++) {
            std::cout << "mset " << i << ":\n";
            mset0[i].Visualize();
            std::cout << "\n";
        }
        std::cout << std::endl;
    }
    return map;
}
