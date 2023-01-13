#include <catch2/catch_all.hpp>

#include "MParT/MapFactory.h"
#include "MParT/MapObjective.h"
#include "MParT/TrainMap.h"

TEST_CASE("Test_TrainMap", "[Test_TrainMap]") {
    unsigned int seed = 155829;
    unsigned int dim = 2;
    unsigned int numPts = 20000;
    unsigned int testPts = numPts / 5;
    auto sampler = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(2);
    sampler->SetSeed(seed);
    auto samples = sampler->Sample(numPts);
    Kokkos::View<double**, Kokkos::HostSpace> targetSamps("targetSamps", 2, numPts);
    Kokkos::parallel_for("Banana", numPts, KOKKOS_LAMBDA(const unsigned int i) {
        targetSamps(0,i) = samples(0,i);
        targetSamps(1,i) = samples(1,i) + samples(0,i)*samples(0,i);
    });
    StridedMatrix<double, Kokkos::HostSpace> testSamps = Kokkos::subview(targetSamps, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(0, testPts));
    StridedMatrix<double, Kokkos::HostSpace> trainSamps = Kokkos::subview(targetSamps, Kokkos::ALL, Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
    auto obj = std::make_shared<KLObjective<Kokkos::HostSpace>>(trainSamps, testSamps, sampler);

    MapOptions map_options;
    map_options.basisType = BasisTypes::ProbabilistHermite;
    auto map = MapFactory::CreateTriangularMap<Kokkos::HostSpace>(dim, dim, 2, map_options);
    TrainOptions train_options;
    train_options.verbose = true;
    TrainMap(map, obj, train_options);
}