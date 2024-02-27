#include "Test_Distributions_Common.h"

#include <numeric>
#include <algorithm>

TEST_CASE( "Testing SampleGenerator", "[SampleGenerator]") {
// Sample 1000 points
// Check empirical CDF against uniform CDF
// assert the difference is less than something
    auto generator = std::make_shared<UniformSampler<Kokkos::HostSpace>>(1);
    unsigned int N_pts = 1000;
    double eps_N = 1e-3 + 1. /std::sqrt(N_pts); // 1/sqrt(N_pts) + epsilon
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
        double euler = std::exp(1.);
        for(unsigned int j = 0; j < N_pts; ++j) {
            double ecdf_j = euler*static_cast<double>(inv_idx[j])/static_cast<double>(N_pts);
            diff += std::abs(ecdf_j - output(0,j));
        }
        diff /= N_pts;
        REQUIRE(diff == Approx(0.).epsilon(eps_N).margin(eps_N));
    }
}
