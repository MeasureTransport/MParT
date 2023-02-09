#include "Test_Distributions_Common.h"

TEST_CASE( "Testing Distribution Class" , "[DistributionClass]") {
    auto sampler = std::make_shared<UniformSampler<Kokkos::HostSpace>>(2);
    auto density = std::make_shared<UniformDensity<Kokkos::HostSpace>>(2);
    auto distribution = Distribution<Kokkos::HostSpace>(sampler, density);
    distribution.SetSeed(42);
    unsigned int N_samp = 5000;
    unsigned int N_bins = 50;
    double hist_tol = 1e-2;

    double bin_inc = std::exp(1.) / N_bins;
    double expected_prop = 1. / (N_bins * N_bins);

    // Create a histogram of the samples
    std::vector<std::vector<unsigned int>> hist(N_bins, std::vector<unsigned int>(N_bins, 0));
    for(int i = 0; i < N_bins; i++) {
        for(int j = 0; j < N_bins; j++) {
            hist[i][j] = 0;
        }
    }

    // Ensure the distribution is 2D
    REQUIRE(distribution.Dim() == 2);

    // Sample from the distribution and fill the histogram
    auto samples = distribution.Sample(N_samp);
    for(int i = 0; i < N_samp; i++) {
        auto x1 = samples(0, i);
        auto x2 = samples(1, i);
        hist[std::floor(x1 / bin_inc)][std::floor(x2 / bin_inc)]++;
    }

    // Check that the histogram is consistent with the expected distribution
    double max_dev = 0.;
    for(int i = 0; i < N_bins; i++) {
        for(int j = 0; j < N_bins; j++) {
            double prop = ((double) hist[i][j]) / ((double) N_samp);
            max_dev = std::max(max_dev, std::abs(prop - expected_prop));
        }
    }
    REQUIRE(max_dev < hist_tol);

    // Check that the density is consistent with the expected distribution
    double density_tol = 1e-6;
    double exp_log_density = -2.;
    double exp_grad_log_density = 0.;
    StridedMatrix<const double, Kokkos::HostSpace> const_samples = samples;
    auto density_samps = distribution.LogDensity(const_samples);
    auto grad_density_samps = distribution.LogDensityInputGrad(const_samples);
    double max_dev_log_density = 0.;
    double max_dev_grad_log_density = 0.;
    for(int i = 0; i < N_samp; i++) {
        max_dev_log_density = std::max(max_dev_log_density, std::abs(density_samps(i) - exp_log_density));
        max_dev_grad_log_density = std::max(max_dev_grad_log_density, std::abs(grad_density_samps(0, i) - exp_grad_log_density));
        max_dev_grad_log_density = std::max(max_dev_grad_log_density, std::abs(grad_density_samps(1, i) - exp_grad_log_density));
    }
    REQUIRE(max_dev_log_density < density_tol);
    REQUIRE(max_dev_grad_log_density < density_tol);
}