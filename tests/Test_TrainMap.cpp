#include <catch2/catch_all.hpp>

#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MapFactory.h"
#include "MParT/MapObjective.h"
#include "MParT/TrainMap.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

// For testing the normality of the pushforward
#include "Distributions/Test_Distributions_Common.h"

using namespace mpart;
using namespace Catch;

#if defined(MPART_ENABLE_GPU)
    using MemorySpace2 = mpart::DeviceSpace;
#else
    using MemorySpace2 = std::false_type;
#endif

TEMPLATE_TEST_CASE("Test_TrainMap", "[TrainMap]", Kokkos::HostSpace, MemorySpace2) {
// Skip the test if the MemorySpace is false_type, i.e., no device space is available
if constexpr (!std::is_same_v<TestType, std::false_type>) {
    using MemorySpace = TestType;
    unsigned int seed = 42;
    unsigned int dim = 2;
    unsigned int numPts = 5000;
    unsigned int testPts = numPts / 5;
    auto sampler = std::make_shared<GaussianSamplerDensity<MemorySpace>>(3);
    sampler->SetSeed(seed);
    auto samples = sampler->Sample(numPts);
    Kokkos::View<double**, MemorySpace> targetSamples("targetSamples", 3, numPts);
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy {0u, numPts};
    Kokkos::parallel_for("Banana", policy, KOKKOS_LAMBDA(const unsigned int i) {
        targetSamples(0,i) = samples(0,i);
        targetSamples(1,i) = samples(1,i);
        targetSamples(2,i) = samples(2,i) + samples(1,i)*samples(1,i);
    });
    unsigned int map_order = 2;
    SECTION("SquareMap") {
        StridedMatrix<const double, MemorySpace> testSamps = Kokkos::subview(targetSamples, Kokkos::make_pair(1u,3u), Kokkos::make_pair(0u, testPts));
        StridedMatrix<const double, MemorySpace> trainSamps = Kokkos::subview(targetSamples, Kokkos::make_pair(1u,3u), Kokkos::make_pair(testPts, numPts));
        auto obj = ObjectiveFactory::CreateGaussianKLObjective(trainSamps, testSamps);

        MapOptions map_options;
        auto map = MapFactory::CreateTriangular<MemorySpace>(dim, dim, map_order, map_options);

        TrainOptions train_options;
        train_options.verbose = 0;
        TrainMap(map, obj, train_options);
        StridedMatrix<double, MemorySpace> pullback_samples = map->Evaluate(testSamps);
        StridedMatrix<double, Kokkos::HostSpace> pullback_samples_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pullback_samples);
        TestStandardNormalSamples(pullback_samples_h);
    }
    SECTION("ComponentMap") {
        StridedMatrix<const double, MemorySpace> testSamps = Kokkos::subview(targetSamples, Kokkos::make_pair(1u,3u), Kokkos::pair<unsigned int, unsigned int>(0, testPts));
        StridedMatrix<const double, MemorySpace> trainSamps = Kokkos::subview(targetSamples, Kokkos::make_pair(1u,3u), Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
        auto obj = ObjectiveFactory::CreateGaussianKLObjective(trainSamps, testSamps, 1);

        MapOptions map_options;
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, map_order);
        FixedMultiIndexSet<MemorySpace> fmset = mset.Fix().ToDevice<MemorySpace>();
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateComponent<MemorySpace>(fmset, map_options);

        TrainOptions train_options;
        train_options.verbose = 0;
        TrainMap(map, obj, train_options);
        StridedMatrix<double, MemorySpace> pullback_samples = map->Evaluate(testSamps);
        StridedMatrix<double, Kokkos::HostSpace> pullback_samples_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pullback_samples);
        TestStandardNormalSamples(pullback_samples_h);
    }
    StridedMatrix<const double, MemorySpace> testSamps = Kokkos::subview(targetSamples, Kokkos::ALL(), Kokkos::pair<unsigned int, unsigned int>(0, testPts));
    StridedMatrix<const double, MemorySpace> trainSamps = Kokkos::subview(targetSamples, Kokkos::ALL(), Kokkos::pair<unsigned int, unsigned int>(testPts, numPts));
    SECTION("RectangleMap") {
        auto obj = ObjectiveFactory::CreateGaussianKLObjective(trainSamps, testSamps, 2);

        MapOptions map_options;
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateTriangular<MemorySpace>(dim+1, dim, map_order, map_options);

        TrainOptions train_options;
        train_options.verbose = 0;
        TrainMap(map, obj, train_options);
        StridedMatrix<double, MemorySpace> pullback_samples = map->Evaluate(testSamps);
        StridedMatrix<double, Kokkos::HostSpace> pullback_samples_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pullback_samples);
        TestStandardNormalSamples(pullback_samples_h);
    }
    SECTION("RectangleMap_reference") {
        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> covar("reference", dim, dim);
        Kokkos::MDRangePolicy<typename MemoryToExecution<MemorySpace>::Space, Kokkos::Rank<2>> policy({0,0}, {dim,dim});
        Kokkos::parallel_for("Covar", policy, KOKKOS_LAMBDA(const unsigned int i, const unsigned int j) {
            covar(i,j) = double(i == j);
        });
        std::shared_ptr<DensityBase<MemorySpace>> dens = std::make_shared<GaussianSamplerDensity<MemorySpace>>(covar); // Ensures that the KLObjective isn't shortcut-ed
        std::shared_ptr<MapObjective<MemorySpace>> obj = std::make_shared<KLObjective<MemorySpace>>(trainSamps, testSamps, dens);
        MapOptions map_options;
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = MapFactory::CreateTriangular<MemorySpace>(dim+1, dim, map_order, map_options);
        TrainOptions train_options;
        train_options.verbose = 0;
        TrainMap(map, obj, train_options);
        StridedMatrix<double, MemorySpace> pullback_samples = map->Evaluate(testSamps);
        StridedMatrix<double, Kokkos::HostSpace> pullback_samples_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pullback_samples);
        TestStandardNormalSamples(pullback_samples_h);
    }
} // End possible skip of GPU test
}