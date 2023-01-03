#include "Test_Distributions_Common.h"

TEST_CASE( "Testing Distribution Class" , "[DistributionClass]") {
    auto generator = std::make_shared<UniformGenerator<Kokkos::HostSpace>>();
    auto density = std::make_shared<UniformDensity<Kokkos::HostSpace>>();
    auto distribution = Distribution<Kokkos::HostSpace>(density, generator);
}