#include "MParT/Distributions/TransportDistributionFactory.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"


using namespace mpart;
using namespace Catch;

TEST_CASE("Creating Pullback/Pushforward distribution", "[CreatePullbackPushforward]") {
    unsigned int dim = 2;
    // Form the affine map T(x) = 2*x + [1,1]
    double diag_el = 2.0;
    double mean = 1.0;
    Kokkos::View<double**, Kokkos::HostSpace> A("A", dim, dim);
    Kokkos::View<double*, Kokkos::HostSpace> b("b", dim);
    for(int i = 0; i < dim; i++) {
        b(i) = mean;
        for(int j = 0; j < dim; j++) {
            A(i, j) = ((double) i == j)*diag_el;
        }
    }

    // Create the map and density to use
    auto map = std::make_shared<AffineMap<Kokkos::HostSpace>>(A, b);
    auto reference = CreateDistribution<Kokkos::HostSpace, GaussianSamplerDensity<Kokkos::HostSpace>>(dim);

    std::shared_ptr<PullbackDistribution<MemorySpace>> pullback = CreatePullback(map, reference);

    std::shared_ptr<PushforwardDistribution<MemorySpace>> pushforward = CreatePushforward(map, reference);
}