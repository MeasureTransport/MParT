#include <catch2/catch_all.hpp>

#include "MParT/MapFactory.h"
#include "MParT/MapObjective.h"
#include "MParT/TrainMap.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

// For testing the normality of the pushforward
#include "Distributions/Test_Distributions_Common.h"

using namespace mpart;
using namespace Catch;

TEST_CASE("Test_TrainMap", "[TrainMap]") {
    unsigned int seed = 42;
    unsigned int dim = 2;
    unsigned int numPts = 20000;
    unsigned int testPts = numPts / 5;
    auto sampler = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(2);
    sampler->SetSeed(seed);
    auto samples = sampler->Sample(numPts);
    Kokkos::View<double**, Kokkos::HostSpace> targetSamples("targetSamples", 2, numPts);
    double max = 0;
    Kokkos::parallel_for("Banana", numPts, KOKKOS_LAMBDA(const unsigned int i) {
        targetSamples(0,i) = samples(0,i);
        targetSamples(1,i) = samples(1,i) + samples(0,i)*samples(0,i);
    });
    StridedMatrix<const double, Kokkos::HostSpace> testSamps = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(0, testPts));
    StridedMatrix<const double, Kokkos::HostSpace> trainSamps = Kokkos::subview(targetSamples, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
    auto obj = ObjectiveFactory::CreateGaussianKLObjective(trainSamps, testSamps);

    MapOptions map_options;
    auto map = MapFactory::CreateTriangular<Kokkos::HostSpace>(dim, dim, 2, map_options);

    TrainOptions train_options;
    train_options.verbose = 0;
    TrainMap(map, obj, train_options);
    auto pullback_samples = map->Evaluate(testSamps);
    TestStandardNormalSamples(pullback_samples);
}