#include<algorithm>
#include <catch2/catch_all.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/DensityBase.h"
#include "MParT/Distributions/PullbackDensity.h"
#include "MParT/Distributions/PushforwardDensity.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/MapFactory.h"
#include "MParT/AffineMap.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing Pullback/Pushforward density", "[PullbackPushforwardDensity]") {
    unsigned int dim = 2;
    unsigned int N_samp = 1000;
    unsigned int seed = 169203;
    SECTION( "AffineMapPullback") {

        // Form the affine map T(x) = 2*x + [1,1]
        double diag_el = 2.0;
        Kokkos::View<double**, Kokkos::HostSpace> A("A", dim, dim);
        Kokkos::View<double*, Kokkos::HostSpace> b("b", dim);
        for(int i = 0; i < dim; i++) {
            b(i) = 1.0;
            for(int j = 0; j < dim; j++) {
                A(i, j) = ((double) i == j)*diag_el;
            }
        }
        // Create the log determinant of the map T analytically
        double logdet = 0.;
        for(int i = 0; i < dim; i++) {
            logdet += std::log(diag_el);
        }

        // Create the map and density to use
        auto map = std::make_shared<AffineMap<Kokkos::HostSpace>>(A, b);
        auto density = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(dim);

        // Create the pullback and pushforward densities
        PullbackDensity<Kokkos::HostSpace> pullback {map, density};
        PushforwardDensity<Kokkos::HostSpace> pushforward {map, density};

        // Set the seed and create samples to test the densities
        density->SetSeed(seed);
        StridedMatrix<const double, Kokkos::HostSpace> samples = density->Sample(N_samp);

        // Initialize the constants for the density calculation
        double offset = -0.9189385332046727; // -log(2*pi)/2
        offset *= dim;

        // Calculate the pullback and pushforward density at the samples
        auto logPullbackDensitySample = pullback.LogDensity(samples);
        auto logPushforwardDensitySample = pushforward.LogDensity(samples);

        // Calculate the pullback and pushforward log density at the samples
        auto gradLogPullbackDensitySample = pullback.LogDensityInputGrad(samples);

        // Ensure that Pushforward::LogDensityInputGrad throws an error
        bool gradLogPushforwardExists = true;
        try {
            pushforward.LogDensityInputGrad(samples);
        } catch (std::runtime_error& e) {
            gradLogPushforwardExists = false;
        }
        REQUIRE(gradLogPushforwardExists == false);

        // Evaluate the map and its inverse at the samples for the analytical calculation
        auto pullbackEvalSample = map->Evaluate(samples);
        Kokkos::View<double**, Kokkos::HostSpace> nullPrefix ("null prefix", 0, N_samp);
        auto pushforwardEvalSample = map->Inverse(nullPrefix, samples);

        Kokkos::fence();
        // Calculate the pullback and pushforward density error
        Kokkos::parallel_for("TestTransportDensity", N_samp, KOKKOS_LAMBDA(const int i) {
            double sampleNormPullback = 0.0;
            double sampleNormPushforward = 0.0;
            for(int j = 0; j < dim; j++) {
                // Take the L2 norm of the pullback and pushforward evaluations
                sampleNormPullback += pullbackEvalSample(j, i)*pullbackEvalSample(j, i);
                sampleNormPushforward += pushforwardEvalSample(j, i)*pushforwardEvalSample(j, i);

                // Check the gradient of the pullback and pushforward density
                gradLogPullbackDensitySample(j, i) -= -pullbackEvalSample(j, i)*diag_el;
            }

            // Calculate the pullback and pushforward density error inplace
            logPullbackDensitySample(i) -= -0.5*sampleNormPullback + offset + logdet;
            logPullbackDensitySample(i) = std::abs(logPullbackDensitySample(i));

            logPushforwardDensitySample(i) -= -0.5*sampleNormPushforward + offset + logdet;
            logPushforwardDensitySample(i) = std::abs(logPushforwardDensitySample(i));
        });
        Kokkos::fence();

        for(int i = 0; i < N_samps; i++) {
            double pullback_err = logPullbackDensitySample(i);
            std::cerr << "logPullbackDensitySample(" << i << ") = " << pullback_err << std::endl;
            CHECK(pullback_err < 1e-6);
        }
        // Check the maximum error
        double max_pullback_err = *std::max_element(logPullbackDensitySample.data(), logPullbackDensitySample.data()+N_samp);
        double max_pushforward_err = *std::max_element(logPushforwardDensitySample.data(), logPushforwardDensitySample.data()+N_samp);

        // Check the maximum gradient error
        double max_pullback_grad_err = *std::max_element(gradLogPullbackDensitySample.data(), gradLogPullbackDensitySample.data()+N_samp*dim);

        REQUIRE(max_pushforward_err < 1e-6);
        REQUIRE(max_pullback_grad_err < 1e-6);
    }

    SECTION( "MapPullbackLogDensityCoeffGrad") {
        // First initialize map
        MapOptions options;
        options.basisType = BasisTypes::ProbabilistHermite;

        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> map = MapFactory::CreateTriangular<Kokkos::HostSpace>(dim,dim,5, options);
        for(int i = 0; i < map->numCoeffs; i++) {
            map->Coeffs()(i) = (double) (i + 1);
        }

        // Create reference and pullback densities
        auto density = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(dim);
        PullbackDensity<Kokkos::HostSpace> pullback {map, density};

        // Create samples
        density->SetSeed(seed);
        StridedMatrix<const double, Kokkos::HostSpace> samples = density->Sample(N_samp);
        Kokkos::View<double**, Kokkos::HostSpace> nullPrefix ("null prefix", 0, N_samp);
        StridedMatrix<const double, Kokkos::HostSpace> pullbackSamples = map->Inverse(nullPrefix, samples);

        // Get appropriate density for the samples plus implemented LogDensityCoeffGrad
        auto samplesDensity = pullback.LogDensity(pullbackSamples);
        StridedMatrix<double, Kokkos::HostSpace> coeffGrad = pullback.LogDensityCoeffGrad(pullbackSamples);

        // Perform first order forward finite difference
        double fdstep = 1e-5;
        for(int i = 0; i < map->numCoeffs; i++) {
            map->Coeffs()(i) += fdstep;
            auto logDensityPerturb = pullback.LogDensity(pullbackSamples);
            Kokkos::parallel_for("Test LogDensityCoeffGrad", N_samp, KOKKOS_LAMBDA(const int j) {
                logDensityPerturb(j) -= samplesDensity(j);
                logDensityPerturb(j) /= fdstep;
                logDensityPerturb(j) -= coeffGrad(i, j);
                logDensityPerturb(j) = std::abs(logDensityPerturb(j));
            });
            Kokkos::fence();
            double max_err = *std::max_element(logDensityPerturb.data(), logDensityPerturb.data()+N_samp);
            REQUIRE(max_err < std::sqrt(fdstep));
            map->Coeffs()(i) = (double) (i + 1);
        }
    }
}