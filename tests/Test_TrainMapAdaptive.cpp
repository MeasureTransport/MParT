#include <catch2/catch_all.hpp>

#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/MapObjective.h"
#include "MParT/TrainMapAdaptive.h"
#include "MParT/MultiIndices/MultiIndex.h"

using namespace mpart;
using namespace Catch;

#include "MParT/Utilities/LinearAlgebra.h"
#include "Distributions/Test_Distributions_Common.h"

void NormalizeSamples(StridedMatrix<double, Kokkos::HostSpace> mat) {
    using MemorySpace = Kokkos::HostSpace;
    unsigned int dim = mat.extent(0);
    unsigned int N_samples = mat.extent(1);
    // Take sum of each row, divide by 1/N_samples
    Kokkos::View<double*, MemorySpace> meanVar("MeanStd", dim);
    for(int i=0; i<dim; ++i){
        for(int j=0; j<N_samples; ++j){
            meanVar(i) += mat(i,j);
        }
        meanVar(i) /= static_cast<double>(N_samples);
    }

    // Subtract mean from each point
    using ExecSpace = typename MemoryToExecution<MemorySpace>::Space;
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>,ExecSpace>({0,0},{dim,N_samples});
    Kokkos::parallel_for("Center data", policy, KOKKOS_LAMBDA(const unsigned int i, const unsigned int j){
        mat(i,j) -= meanVar(i);
    });

    // Take || . ||_2^2 of each row, divide by 1/(N_samples-1)
    for(int i=0; i<dim; ++i){
        meanVar(i) = 0.0;
        for(int j=0; j<N_samples; ++j){
            meanVar(i) += mat(i,j)*mat(i,j);
        }
        meanVar(i) /= static_cast<double>(N_samples-1);
    }

    // Divide each point by appropriate standard deviation
    Kokkos::parallel_for("Scale data", policy, KOKKOS_LAMBDA(const unsigned int i, const unsigned int j){
        mat(i,j) /= std::sqrt(meanVar(i));
    });
    Kokkos::fence("Normalize data");
}

TEST_CASE("Adaptive Transport Map","[ATM]") {
    unsigned int dim = 2;
    unsigned int seed = 13982;
    unsigned int numPts = 20000;
    unsigned int testPts = numPts / 5;

    auto g_sampler = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(dim);
    g_sampler->SetSeed(seed);
    StridedMatrix<double,Kokkos::HostSpace> samples = g_sampler->Sample(numPts);
    // SECTION("ModifiedBanana") {
    //     Kokkos::View<double**, Kokkos::HostSpace> targetSamples("targetSamples", 2, numPts);
    //     Kokkos::parallel_for("Intializing targetSamples", numPts, KOKKOS_LAMBDA(const unsigned int i){
    //         targetSamples(0,i) = samples(0,i);
    //         targetSamples(1,i) = samples(1,i) + samples(0,i) + samples(0,i)*samples(0,i);
    //     });
    //     NormalizeSamples(targetSamples);

    //     StridedMatrix<const double, Kokkos::HostSpace> testSamples = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(0, testPts));
    //     StridedMatrix<const double, Kokkos::HostSpace> trainSamples = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
    //     auto objective = ObjectiveFactory::CreateGaussianKLObjective(trainSamples,testSamples);

    //     std::vector<MultiIndexSet> mset0 {MultiIndexSet::CreateTotalOrder(1,0), MultiIndexSet::CreateTotalOrder(2,0)};
    //     MultiIndexSet correctMset1 = MultiIndexSet::CreateTotalOrder(1,1);
    //     MultiIndexSet correctMset2 = MultiIndexSet::CreateTotalOrder(2,1) + MultiIndex{2,0};

    //     ATMOptions opts;
    //     opts.maxSize = 8; // Algorithm must add 4 correct terms in 6 iterations
    //     opts.basisLB = -3.;
    //     opts.basisUB = 3.;

    //     std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> atm = TrainMapAdaptive<Kokkos::HostSpace>(mset0, objective, opts);
    //     MultiIndexSet finalMset1 = mset0[0];
    //     MultiIndexSet finalMset2 = mset0[1];
    //     CHECK((finalMset1 + correctMset1).Size() == finalMset1.Size());
    //     CHECK((finalMset2 + correctMset2).Size() == finalMset2.Size());
    //     StridedMatrix<double, Kokkos::HostSpace> pullback_test = atm->Evaluate(testSamples);
    //     TestStandardNormalSamples(pullback_test);
    // }
    // SECTION("GaussianMixture") {
    //     double mean_1=-2,std_1=std::sqrt(0.5);
    //     double mean_2= 2,std_2=std::sqrt(  2);
    //     double prob_1 = 0.5;
    //     auto u_sampler = std::make_shared<UniformSampler<Kokkos::HostSpace>>(1,1.);
    //     StridedMatrix<double,Kokkos::HostSpace> u_samples = u_sampler->Sample(numPts);
    //     StridedMatrix<double,Kokkos::HostSpace> targetSamples = Kokkos::View<double**,Kokkos::HostSpace>("targetSamples",2,numPts);
    //     Kokkos::parallel_for("Create Gaussian mixture", numPts, KOKKOS_LAMBDA(const unsigned int i){
    //         bool pick_1 = u_samples(0,i) < prob_1;
    //         double scale = pick_1 ? std_1 : std_2;
    //         double shift = pick_1 ? mean_1 : mean_2;
    //         for(int j = 0; j < dim; j++) {
    //             targetSamples(j,i) = samples(j,i)*scale;
    //             samples(j,i) += shift;
    //         }
    //     });
    //     NormalizeSamples(targetSamples);
    //     StridedMatrix<const double, Kokkos::HostSpace> testSamples = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(0, testPts));
    //     StridedMatrix<const double, Kokkos::HostSpace> trainSamples = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
    //     auto objective = ObjectiveFactory::CreateGaussianKLObjective(trainSamples,testSamples);

    //     ATMOptions opts;
    //     opts.maxSize = 5;
    //     opts.basisLB = -3.;
    //     opts.basisUB = 3.;

    //     std::vector<MultiIndexSet> mset0 {MultiIndexSet::CreateTotalOrder(1,0), MultiIndexSet::CreateTotalOrder(2,0)};
    //     std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> atm = TrainMapAdaptive<Kokkos::HostSpace>(mset0, objective, opts);
    //     StridedMatrix<double, Kokkos::HostSpace> pullback_test = atm->Evaluate(testSamples);
    //     TestStandardNormalSamples(pullback_test);
    // }
    // SECTION("TraditionalBanana") {
    //     Kokkos::View<double**, Kokkos::HostSpace> targetSamples("targetSamples", 2, numPts);
    //     Kokkos::parallel_for("Intializing targetSamples", numPts, KOKKOS_LAMBDA(const unsigned int i){
    //         targetSamples(0,i) = samples(0,i);
    //         targetSamples(1,i) = samples(1,i) + samples(0,i)*samples(0,i);
    //     });
    //     NormalizeSamples(targetSamples);

    //     StridedMatrix<const double, Kokkos::HostSpace> testSamples = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(0, testPts));
    //     StridedMatrix<const double, Kokkos::HostSpace> trainSamples = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
    //     auto objective = ObjectiveFactory::CreateGaussianKLObjective(trainSamples,testSamples);

    //     std::vector<MultiIndexSet> mset0 {MultiIndexSet::CreateTotalOrder(1,0), MultiIndexSet::CreateTotalOrder(2,0)};
    //     MultiIndexSet correctMset1 = MultiIndexSet::CreateTotalOrder(1,1);
    //     MultiIndexSet correctMset2 = (MultiIndexSet::CreateTotalOrder(2,0) + MultiIndex{0,1}) + MultiIndex{2,0};

    //     ATMOptions opts;
    //     opts.maxSize = 15; // Algorithm must add 3 correct terms in 6 iterations
    //     opts.basisLB = -3.;
    //     opts.basisUB = 3.;
    //     opts.maxDegrees = MultiIndex{1000,4}; // Limit the second input to have cubic complexity or less

    //     std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> atm = TrainMapAdaptive<Kokkos::HostSpace>(mset0, objective, opts);
    //     MultiIndexSet finalMset1 = mset0[0];
    //     MultiIndexSet finalMset2 = mset0[1];
    //     CHECK((finalMset1 + correctMset1).Size() == finalMset1.Size());
    //     CHECK((finalMset2 + correctMset2).Size() == finalMset2.Size());
    //     std::vector<bool> bounded1 = finalMset1.FilterBounded(opts.maxDegrees);
    //     std::vector<bool> bounded2 = finalMset2.FilterBounded(opts.maxDegrees);
    //     bool checkBound = false;
    //     for(auto b1 : bounded1) checkBound |= b1;
    //     for(auto b2 : bounded2) checkBound |= b2;
    //     CHECK(!checkBound);
    //     StridedMatrix<double, Kokkos::HostSpace> pullback_test = atm->Evaluate(testSamples);
    //     TestStandardNormalSamples(pullback_test);
    // }
    SECTION("TraditionalBananaOneComp") {
        Kokkos::View<double**, Kokkos::HostSpace> targetSamples("targetSamples", 2, numPts);
        Kokkos::parallel_for("Intializing targetSamples", numPts, KOKKOS_LAMBDA(const unsigned int i){
            targetSamples(0,i) = samples(0,i);
            targetSamples(1,i) = samples(1,i) + samples(0,i)*samples(0,i);
        });
        NormalizeSamples(targetSamples);

        StridedMatrix<const double, Kokkos::HostSpace> testSamples = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(0, testPts));
        StridedMatrix<const double, Kokkos::HostSpace> trainSamples = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
        auto objective = ObjectiveFactory::CreateGaussianKLObjective(trainSamples,testSamples,1);

        std::vector<MultiIndexSet> mset0 {MultiIndexSet::CreateTotalOrder(2,0)};
        MultiIndexSet correctMset = (MultiIndexSet::CreateTotalOrder(2,0) + MultiIndex{0,1}) + MultiIndex{2,0};

        ATMOptions opts;
        opts.maxSize = 15; // Algorithm must add 3 correct terms in 6 iterations
        opts.basisLB = -3.;
        opts.basisUB = 3.;
        opts.maxDegrees = MultiIndex{1000,4}; // Limit the second input to have cubic complexity or less

        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> atm = TrainMapAdaptive<Kokkos::HostSpace>(mset0, objective, opts);
        MultiIndexSet finalMset = mset0[0];
        CHECK((finalMset + correctMset).Size() == finalMset.Size());
        std::vector<bool> bounded = finalMset.FilterBounded(opts.maxDegrees);
        bool checkBound = false;
        for(auto b1 : bounded) checkBound |= b1;
        CHECK(!checkBound);
        StridedMatrix<double, Kokkos::HostSpace> pullback_test = atm->Evaluate(testSamples);
        TestStandardNormalSamples(pullback_test);
    }
}