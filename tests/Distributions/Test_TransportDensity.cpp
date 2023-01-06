#include<algorithm>
#include <catch2/catch_all.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/DensityBase.h"
#include "MParT/Distributions/PullbackDensity.h"
#include "MParT/Distributions/PushforwardDensity.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/AffineMap.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing Pullback/Pushforward density", "[PullbackPushforwardDensity]") {
    unsigned int dim = 2;
    unsigned int N_samp = 1000;

    double diag_el = 2.0;
    Kokkos::View<double**, Kokkos::HostSpace> A("A", dim, dim);
    Kokkos::View<double*, Kokkos::HostSpace> b("b", dim);
    for(int i = 0; i < dim; i++) {
        b(i) = 1.0;
        for(int j = 0; j < dim; j++) {
            A(i, j) = ((double) i == j)*diag_el;
        }
    }
    double logdet = 0.;
    for(int i = 0; i < dim; i++) {
        logdet += std::log(diag_el);
    }
    auto map = std::make_shared<AffineMap<Kokkos::HostSpace>>(A, b);
    auto density = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(dim);
    PullbackDensity<Kokkos::HostSpace> pullback (map, density);
    PushforwardDensity<Kokkos::HostSpace> pushforward {map, density};
    density->SetSeed(0);
    StridedMatrix<const double, Kokkos::HostSpace> samples = density->Sample(N_samp);
    double offset = -0.9189385332046727; // -log(2*pi)/2
    offset *= dim;
    auto logPullbackDensitySample = pullback.LogDensity(samples);
    auto logPushforwardDensitySample = pushforward.LogDensity(samples);
    auto pullbackEvalSample = map->Evaluate(samples);
    Kokkos::View<double**, Kokkos::HostSpace> nullPrefix ("null prefix", 0, N_samp);
    auto pushforwardEvalSample = map->Inverse(nullPrefix, samples);
    Kokkos::parallel_for("TestPullbackDensity", N_samp, KOKKOS_LAMBDA(const int i) {
        // Take the norm of the pullback and pushforward evaluations
        double sampleNormPullback = 0.0;
        double sampleNormPushforward = 0.0;
        for(int j = 0; j < dim; j++) {
            sampleNormPullback += pullbackEvalSample(j, i)*pullbackEvalSample(j, i);
            sampleNormPushforward += pushforwardEvalSample(j, i)*pushforwardEvalSample(j, i);
        }

        // Calculate the pullback and pushforward density error inplace
        logPullbackDensitySample(i) -= -0.5*sampleNormPullback + offset + logdet;
        logPullbackDensitySample(i) = std::abs(logPullbackDensitySample(i));

        logPushforwardDensitySample(i) -= -0.5*sampleNormPushforward + offset + logdet;
        logPushforwardDensitySample(i) = std::abs(logPushforwardDensitySample(i));
    });
    double max_pullback_err = *std::max_element(logPullbackDensitySample.data(), logPullbackDensitySample.data()+N_samp);
    double max_pushforward_err = *std::max_element(logPushforwardDensitySample.data(), logPushforwardDensitySample.data()+N_samp);

    REQUIRE(max_pullback_err < 1e-6);
    REQUIRE(max_pushforward_err < 1e-6);
}