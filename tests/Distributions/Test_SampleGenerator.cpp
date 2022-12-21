#include<algorithm>
#include <catch2/catch_all.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/SampleGenerator.h"

using namespace mpart;
using namespace Catch;


// Uniform generator on [0,2]
class UniformGenerator {
public:

using PoolType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
PoolType rand_pool;

// Set a given seed for this test
UniformGenerator(): rand_pool(160258) {}

void SampleImpl(Kokkos::View<double**, Kokkos::HostSpace> output) {
    unsigned int N = output.extent(1);
    Kokkos::parallel_for("uniform generator", N, KOKKOS_LAMBDA(const int& j) {
        typename PoolType::generator_type rgen = rand_pool.get_state();
        output(0,j) = 2*rgen.drand();
        rand_pool.free_state(rgen);
    });
}
};

TEST_CASE( "Testing SampleGenerator", "[SampleGenerator]") {
// Sample 1000 points
// Check empirical CDF against uniform CDF
// assert the difference is less than something
    auto generator = std::make_shared<UniformGenerator>();
    unsigned int N_pts = 1000;
    SECTION("SampleImpl") {
        Kokkos::View<double**, Kokkos::HostSpace> output ("output", 1, N_pts);
        generator->SampleImpl(output);
        std::vector<int> idx (N_pts);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&output](int i1, int i2) {return output(0,i1) < output(0,i2);});

        // Need to take inverse permutation to get correct ordering for exact CDF calculation
        std::vector<int> inv_idx (N_pts);
        for(unsigned int j = 0; j < N_pts; ++j) {
            inv_idx[idx[j]] = j;
        }

        // Calculate mean L1 distance between empirical CDF and uniform CDF
        double diff = 0.;
        for(unsigned int j = 0; j < N_pts; ++j) {
            double ecdf_j = 2*static_cast<double>(inv_idx[j])/static_cast<double>(N_pts);
            diff += std::abs(ecdf_j - output(0,j));
        }
        diff /= N_pts;
        REQUIRE(diff == Approx(0.).epsilon(1e-2).margin(1e-2));
    }
}
