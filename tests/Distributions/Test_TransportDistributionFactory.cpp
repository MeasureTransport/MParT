#include <catch2/catch_all.hpp>
#include "MParT/AffineMap.h"
#include "MParT/Distributions/TransportDistributionFactory.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"


using namespace mpart;
using namespace mpart::TransportDistributionFactory;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE("Creating Pullback/Pushforward distribution", "[CreatePullbackPushforward]") {
    unsigned int dim = 2;
    // Form the affine map T(x) = 2*x + [1,1]
    double diag_el = 2.0;
    double mean = 1.0;
    Kokkos::View<double**, MemorySpace> A("A", dim, dim);
    Kokkos::View<double*, MemorySpace> b("b", dim);
    for(int i = 0; i < dim; i++) {
        b(i) = mean;
        for(int j = 0; j < dim; j++) {
            A(i, j) = ((double) i == j)*diag_el;
        }
    }

    // Create the map and density to use
    auto map = std::make_shared<AffineMap<MemorySpace>>(A, b);

    SECTION("[GeneralDistributionConstruction]"){
        auto reference = CreateDistribution<MemorySpace, GaussianSamplerDensity<MemorySpace>>(dim);
        std::shared_ptr<Distribution<MemorySpace>> pullback = CreatePullback<MemorySpace>(map, reference);
        std::shared_ptr<Distribution<MemorySpace>> pushforward = CreatePushforward<MemorySpace>(map, reference);
    }
    SECTION("Testing Gaussian shortcut") {
        std::shared_ptr<Distribution<MemorySpace>> pullback = CreateGaussianPullback<MemorySpace>(map);
        std::shared_ptr<Distribution<MemorySpace>> pushforward = CreateGaussianPushforward<MemorySpace>(map);
    }
}