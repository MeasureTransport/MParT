#include<algorithm>
#include <catch2/catch_all.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/PullbackSampler.h"
#include "MParT/Distributions/PushforwardSampler.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/AffineMap.h"

#include "Test_Distributions_Common.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing Pullback/Pushforward sampling", "[PullbackPushforwardSampler]") {
    unsigned int dim = 2;
    unsigned int N_samp = 10000;
    unsigned int seed = 162849;

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
    auto density = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(dim);

    SECTION("PullbackSampler") {
        PullbackSampler<Kokkos::HostSpace> pullback {map, density};

        // Set the seed and create samples to test the densities
        pullback.SetSeed(seed);

        auto pullbackSamples = pullback.Sample(N_samp);

        // Calculate the pullback and pushforward density error
        auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_samp, dim});
        Kokkos::parallel_for("Normalize Pullback Samples", policy, KOKKOS_LAMBDA(const int j, const int i) {
            pullbackSamples(i,j) *= diag_el;
            pullbackSamples(i,j) += mean;
        });

        TestStandardNormalSamples(pullbackSamples);
    }
    SECTION("PushforwardSampler") {
        // Create the pushforward sampler
        PushforwardSampler<Kokkos::HostSpace> pushforward {map, density};

        // Set the seed and create samples
        pushforward.SetSeed(seed);
        auto pushforwardSamples = pushforward.Sample(N_samp);

        // Calculate the pullback and pushforward density error
        auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {N_samp, dim});
        Kokkos::parallel_for("Normalize Samples", policy, KOKKOS_LAMBDA(const int j, const int i) {
            pushforwardSamples(i,j) -= mean;
            pushforwardSamples(i,j) /= diag_el;
        });

        TestStandardNormalSamples(pushforwardSamples);
    }
}