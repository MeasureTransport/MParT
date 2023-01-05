#include <algorithm>
#include <catch2/catch_all.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing Gaussian Distribution", "[GaussianDist]") {
    unsigned int dim = 3;
    unsigned int N_samp = 1000;
    unsigned int N_bins = 25;
    double mc_margin = (1/std::sqrt(N_samp))*3.0;
    double abs_margin = 1e-6;

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
            REQUIRE(mean(i) == Approx(0.0).margin(mc_margin));
            for(int j = 0; j < dim; j++) {
                double diag = (double) (i == j);
                REQUIRE(covar(i, j) - mean(i)*mean(j) == Approx(diag).margin(mc_margin));
            }
        }

        std::vector<unsigned int> in_one_std (dim, 0);
        std::vector<unsigned int> in_two_std (dim, 0);
        std::vector<unsigned int> in_three_std (dim, 0);
        for(int i = 0; i < N_samp; i++) {
            for(int j = 0; j < dim; j++) {
                double samp_abs = std::abs(samples(j, i));
                if(samp_abs < 1.0) {
                    in_one_std[j]++;
                }
                if(samp_abs < 2.0) {
                    in_two_std[j]++;
                }
                if(samp_abs < 3.0) {
                    in_three_std[j]++;
                }
            }
        }
        double emp_one_std = 0.682689492137;
        double emp_two_std = 0.954499736104;
        double emp_three_std = 0.997300203937;
        for(int i = 0; i < dim; i++) {
            REQUIRE(in_one_std[i]/(double)N_samp == Approx(emp_one_std).margin(mc_margin));
            REQUIRE(in_two_std[i]/(double)N_samp == Approx(emp_two_std).margin(mc_margin));
            REQUIRE(in_three_std[i]/(double)N_samp == Approx(emp_three_std).margin(mc_margin));
        }

        StridedVector<double, Kokkos::HostSpace> samples_pdf = dist.LogDensity(samples);
        double offset = -0.9189385332046728*dim; // -log(2*pi)*dim/2
        // Calculate difference of samples_pdf and the true pdf in place
        Kokkos::parallel_for(N_samp, KOKKOS_LAMBDA(const int i) {
            double norm = 0.;
            for(int j = 0; j < dim; j++) {
                norm += samples(j, i)*samples(j, i);
            }
            samples_pdf(i) -= offset - 0.5*norm;
            samples_pdf(i) = std::abs(samples_pdf(i));
        });
        // Find the maximum difference and assert it's within the margin of error
        double max_pdf = *std::max_element(samples_pdf.data(), samples_pdf.data()+N_samp);
        REQUIRE(max_pdf == Approx(0.0).margin(abs_margin));
    }

    SECTION( "Default Covariance, unit mean in all dimensions" ) {

    }
    SECTION( "Diagonal Covariance, Default mean" ) {

    }
    SECTION( "Diagonal Covariance, unit mean in all dimensions" ) {

    }
}