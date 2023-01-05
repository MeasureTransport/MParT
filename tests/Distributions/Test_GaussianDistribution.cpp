#include <algorithm>
#include <catch2/catch_all.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

using namespace mpart;
using namespace Catch;

// Tests samples that should be transformed to a standard normal distribution
void TestStandardNormalSamples(StridedMatrix<double, Kokkos::HostSpace> samples) {
    unsigned int dim = samples.extent(0);
    unsigned int N_samp = samples.extent(1);
    double mc_margin = (1/std::sqrt(N_samp))*3.0;

    GaussianDistribution<Kokkos::HostSpace> dist = CreateDistribution<Kokkos::HostSpace, GaussianSamplerDensity<Kokkos::HostSpace>>(dim);
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
}

// Tests samples that should be transformed to a standard normal distribution and the pdf of the samples prior to transformation
void TestGaussianLogPDF(StridedMatrix<double, Kokkos::HostSpace> samples, StridedVector<double, Kokkos::HostSpace> samples_pdf,
        double logdet_cov, double abs_margin) {
    unsigned int dim = samples.extent(0);
    unsigned int N_samp = samples.extent(1);

    double offset = -0.9189385332046728*dim; // -log(2*pi)*dim/2
    offset -= 0.5*logdet_cov;
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

TEST_CASE( "Testing Gaussian Distribution", "[GaussianDist]") {
    unsigned int dim = 3;
    unsigned int N_samp = 1000;
    double covar_diag_val = 4.0;
    double abs_margin = 1e-6;

    SECTION( "Default Covariance, Default mean" ) {
        GaussianDistribution<Kokkos::HostSpace> dist = CreateDistribution<Kokkos::HostSpace, GaussianSamplerDensity<Kokkos::HostSpace>>(dim);
        StridedMatrix<double, Kokkos::HostSpace> samples = dist.Sample(N_samp);
        StridedVector<double, Kokkos::HostSpace> samples_pdf = dist.LogDensity(samples);
        TestStandardNormalSamples(samples);
        TestGaussianLogPDF(samples, samples_pdf, 0., abs_margin);
    }


    Kokkos::View<double*, Kokkos::HostSpace> mean("mean", dim);
    std::fill(mean.data(), mean.data()+dim, 1.0);

    SECTION( "Default Covariance, unit mean in all dimensions" ) {
        GaussianDistribution<Kokkos::HostSpace> dist = CreateDistribution<Kokkos::HostSpace, GaussianSamplerDensity<Kokkos::HostSpace>>(mean);
        StridedMatrix<double, Kokkos::HostSpace> samples = dist.Sample(N_samp);
        StridedVector<double, Kokkos::HostSpace> samples_pdf = dist.LogDensity(samples);
        Kokkos::parallel_for(dim, KOKKOS_LAMBDA(const int i) {
            for(int j = 0; j < N_samp; j++) {
                samples(i, j) -= 1.0;
            }
        });
        TestStandardNormalSamples(samples);
        TestGaussianLogPDF(samples, samples_pdf, 0., abs_margin);
    }

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> covar("covar", dim, dim);
    auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {dim, dim});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i, const int j) {
        covar(i, j) = ((double) (i == j))*covar_diag_val;
    });

    SECTION( "Diagonal Covariance, Default mean" ) {
        GaussianDistribution<Kokkos::HostSpace> dist = CreateDistribution<Kokkos::HostSpace, GaussianSamplerDensity<Kokkos::HostSpace>>(covar);
        StridedMatrix<double, Kokkos::HostSpace> samples = dist.Sample(N_samp);
        StridedVector<double, Kokkos::HostSpace> samples_pdf = dist.LogDensity(samples);
        policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {dim, N_samp});
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i, const int j) {
            samples(i, j) /= 2.;
        });
        TestStandardNormalSamples(samples);
        TestGaussianLogPDF(samples, samples_pdf, dim*std::log(covar_diag_val), abs_margin);
    }

    SECTION( "Diagonal Covariance, unit mean in all dimensions" ) {
        GaussianDistribution<Kokkos::HostSpace> dist = CreateDistribution<Kokkos::HostSpace, GaussianSamplerDensity<Kokkos::HostSpace>>(mean, covar);
        StridedMatrix<double, Kokkos::HostSpace> samples = dist.Sample(N_samp);
        StridedVector<double, Kokkos::HostSpace> samples_pdf = dist.LogDensity(samples);
        policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {dim, N_samp});
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i, const int j) {
            samples(i, j) /= 2.;
            samples(i, j) -= 1.0;
        });
        TestStandardNormalSamples(samples);
        TestGaussianLogPDF(samples, samples_pdf, dim*std::log(covar_diag_val), abs_margin);
    }
}