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
    unsigned int numPts = 5000;
    unsigned int testPts = numPts / 5;
    auto sampler = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(3);
    sampler->SetSeed(seed);
    auto samples = sampler->Sample(numPts);
    Kokkos::View<double**, Kokkos::HostSpace> targetSamples("targetSamples", 3, numPts);
    double max = 0;
    Kokkos::parallel_for("Banana", numPts, KOKKOS_LAMBDA(const unsigned int i) {
        targetSamples(0,i) = samples(0,i);
        targetSamples(1,i) = samples(1,i);
        targetSamples(2,i) = samples(2,i) + samples(1,i)*samples(1,i);
    });
    unsigned int map_order = 2;
    SECTION("SquareMap") {
        StridedMatrix<const double, Kokkos::HostSpace> testSamps = Kokkos::subview(targetSamples, Kokkos::make_pair(1u,3u), Kokkos::make_pair(0u, testPts));
        StridedMatrix<const double, Kokkos::HostSpace> trainSamps = Kokkos::subview(targetSamples, Kokkos::make_pair(1u,3u), Kokkos::make_pair(testPts, numPts));
        auto obj = ObjectiveFactory::CreateGaussianKLObjective(trainSamps, testSamps);

        MapOptions map_options;
        auto map = MapFactory::CreateTriangular<Kokkos::HostSpace>(dim, dim, map_order, map_options);

        TrainOptions train_options;
        train_options.verbose = 0;
        TrainMap(map, obj, train_options);
        auto pullback_samples = map->Evaluate(testSamps);
        TestStandardNormalSamples(pullback_samples);
    }
    SECTION("ComponentMap") {
        StridedMatrix<const double, Kokkos::HostSpace> testSamps = Kokkos::subview(targetSamples, Kokkos::make_pair(1u,3u), Kokkos::pair<unsigned int, unsigned int>(0, testPts));
        StridedMatrix<const double, Kokkos::HostSpace> trainSamps = Kokkos::subview(targetSamples, Kokkos::make_pair(1u,3u), Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
        auto obj = ObjectiveFactory::CreateGaussianKLObjective(trainSamps, testSamps, 1);

        MapOptions map_options;
        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map = MapFactory::CreateComponent<Kokkos::HostSpace>(FixedMultiIndexSet<Kokkos::HostSpace>(dim,map_order), map_options);

        TrainOptions train_options;
        train_options.verbose = 0;
        TrainMap(map, obj, train_options);
        auto pullback_samples = map->Evaluate(testSamps);
        TestStandardNormalSamples(pullback_samples);
    }
    SECTION("RectangleMap") {
        StridedMatrix<const double, Kokkos::HostSpace> testSamps = Kokkos::subview(targetSamples, Kokkos::ALL(), Kokkos::pair<unsigned int, unsigned int>(0, testPts));
        StridedMatrix<const double, Kokkos::HostSpace> trainSamps = Kokkos::subview(targetSamples, Kokkos::ALL(), Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
        auto obj = ObjectiveFactory::CreateGaussianKLObjective(trainSamps, testSamps, 2);

        MapOptions map_options;
        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map = MapFactory::CreateTriangular<Kokkos::HostSpace>(dim+1, dim, map_order, map_options);

        TrainOptions train_options;
        train_options.verbose = 0;
        TrainMap(map, obj, train_options);
        auto pullback_samples = map->Evaluate(testSamps);
        TestStandardNormalSamples(pullback_samples);
    }
}