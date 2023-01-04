#include "Test_Distributions_Common.h"

TEST_CASE( "Testing Distribution Class" , "[DistributionClass]") {
    auto sampler = std::make_shared<UniformSampler<Kokkos::HostSpace>>(2);
    auto density = std::make_shared<UniformDensity<Kokkos::HostSpace>>();
    auto distribution = Distribution<Kokkos::HostSpace>(sampler, density);

}