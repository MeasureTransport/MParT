#include <algorithm>
#include <catch2/catch_all.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing Gaussian Distribution", "[GaussianDist]") {
    unsigned int dim = 3;
    unsigned int N_samp = 5000;
    double mc_margin = 0.1;
    SECTION( "Default Covariance, Default mean" ) {
        GaussianDistribution<Kokkos::HostSpace> dist = CreateDistribution<Kokkos::HostSpace, GaussianSamplerDensity<Kokkos::HostSpace>>(dim);
        auto samples = dist.Sample(N_samp);
        Kokkos::View<double*, Kokkos::HostSpace> mean("mean", dim);
        Kokkos::View<double**, Kokkos::HostSpace> covar("covar", dim, dim);
        std::fill(mean.data(), mean.data()+dim, 0.0);
        std::fill(covar.data(), covar.data()+dim*dim, 0.0);
        // Calculate sample mean and sample covariance
        for(int i = 0; i < N_samp; i++) {
            for(int j = 0; j < dim; j++) {
                mean(j) += samples(j, i)/N_samp;
                for(int k = 0; k < dim; k++) {
                    covar(j, k) += samples(j, i) * samples(k, i)/(N_samp-1);
                }
            }
        }
        // Check that the mean is zero and the covariance is the identity matrix
        for(int i = 0; i < dim; i++) {
            for(int j = 0; j < dim; j++) {
                if(i == j) {
                    REQUIRE(mean(i) == Approx(0.0).margin(mc_margin));
                    REQUIRE(covar(i, j) - mean(i)*mean(j) == Approx(1.0).margin(mc_margin));
                } else {
                    REQUIRE(covar(i, j) - mean(i)*mean(j) == Approx(0.0).margin(mc_margin));
                }
            }
        }
    }
    SECTION( "Default Covariance, unit mean in all dimensions" ) {

    }
    SECTION( "Diagonal Covariance, Default mean" ) {

    }
    SECTION( "Diagonal Covariance, unit mean in all dimensions" ) {

    }
}