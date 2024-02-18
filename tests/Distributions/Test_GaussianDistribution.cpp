#include <algorithm>
#include <catch2/catch_all.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "Test_Distributions_Common.h"
#include <iostream>

using namespace mpart;
using namespace Catch;

// Tests samples that should be transformed to a standard normal distribution and the pdf of the samples prior to transformation
void TestGaussianLogPDF(StridedMatrix<double, Kokkos::HostSpace> samples, StridedVector<double, Kokkos::HostSpace> samples_pdf,
        StridedMatrix<double, Kokkos::HostSpace> samples_gradpdf, double logdet_cov, double sqrt_diag, double abs_margin) {
    unsigned int dim = samples.extent(0);
    unsigned int N_samp = samples.extent(1);

    double offset = -0.9189385332046728*dim; // -log(2*pi)*dim/2
    offset -= 0.5*logdet_cov;

    Kokkos::RangePolicy<typename MemoryToExecution<Kokkos::HostSpace>::Space> policy {0, N_samp};
    // Calculate difference of samples_pdf and the true pdf in place
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
        double norm = 0.;
        for(int j = 0; j < dim; j++) {
            norm += samples(j, i)*samples(j, i);
            samples_gradpdf(j,i) += samples(j, i)/sqrt_diag;
            samples_gradpdf(j,i) = std::abs(samples_gradpdf(j,i));
        }
        samples_pdf(i) -= offset - 0.5*norm;
        samples_pdf(i) = std::abs(samples_pdf(i));
    });
    // Find the maximum difference and assert it's within the margin of error
    double max_pdf_err = *std::max_element(samples_pdf.data(), samples_pdf.data()+N_samp);
    double max_gradpdf_err = *std::max_element(samples_gradpdf.data(), samples_gradpdf.data()+N_samp*dim);
    REQUIRE(max_pdf_err < abs_margin);
    REQUIRE(max_gradpdf_err < abs_margin);
}

TEST_CASE( "Testing Gaussian Distribution", "[GaussianDist]") {
    unsigned int dim = 3;
    unsigned int N_samp = 5000;
    double covar_diag_val = 4.0;
    double abs_margin = 1e-6;
    unsigned int seed = 162849;

    SECTION( "Default Covariance, Default mean" ) {
        std::shared_ptr<Distribution<Kokkos::HostSpace>> dist = CreateGaussianDistribution<Kokkos::HostSpace>(dim);
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> samples ("sample matrix", dim, N_samp);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> samples_pdf ("sample pdf", N_samp);
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> samples_gradpdf ("sample grad pdf", dim, N_samp);
        dist->SetSeed(seed);
        dist->SampleImpl(samples);
        dist->LogDensityImpl(samples, samples_pdf);
        dist->LogDensityInputGradImpl(samples, samples_gradpdf);
        TestStandardNormalSamples(samples);
        TestGaussianLogPDF(samples, samples_pdf, samples_gradpdf, 0., 1., abs_margin);
    }


    Kokkos::View<double*, Kokkos::HostSpace> mean("mean", dim);
    std::fill(mean.data(), mean.data()+dim, 1.0);

    SECTION( "Default Covariance, unit mean in all dimensions" ) {
        std::shared_ptr<Distribution<Kokkos::HostSpace>> dist = CreateGaussianDistribution<Kokkos::HostSpace>(mean);
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> samples ("sample matrix", dim, N_samp);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> samples_pdf ("sample pdf", N_samp);
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> samples_gradpdf ("sample grad pdf", dim, N_samp);
        dist->SetSeed(seed);
        dist->SampleImpl(samples);
        dist->LogDensityImpl(samples, samples_pdf);
        dist->LogDensityInputGradImpl(samples, samples_gradpdf);
        Kokkos::RangePolicy<typename MemoryToExecution<Kokkos::HostSpace>::Space> policy {0, dim};
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i) {
            for(int j = 0; j < N_samp; j++) {
                samples(i, j) -= 1.0;
            }
        });
        TestStandardNormalSamples(samples);
        TestGaussianLogPDF(samples, samples_pdf, samples_gradpdf, 0., 1., abs_margin);
    }

    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> covar("covar", dim, dim);
    Kokkos::MDRangePolicy<typename MemoryToExecution<Kokkos::HostSpace>::Space, Kokkos::Rank<2>> policy ({0, 0}, {dim, dim});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i, const int j) {
        covar(i, j) = ((double) (i == j))*covar_diag_val;
    });

    SECTION( "Diagonal Covariance, Default mean" ) {
        std::shared_ptr<Distribution<Kokkos::HostSpace>> dist = CreateGaussianDistribution<Kokkos::HostSpace>(covar);
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> samples ("sample matrix", dim, N_samp);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> samples_pdf ("sample pdf", N_samp);
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> samples_gradpdf ("sample grad pdf", dim, N_samp);
        dist->SetSeed(seed);
        dist->SampleImpl(samples);
        dist->LogDensityImpl(samples, samples_pdf);
        dist->LogDensityInputGradImpl(samples, samples_gradpdf);
        policy = Kokkos::MDRangePolicy<typename MemoryToExecution<Kokkos::HostSpace>::Space, Kokkos::Rank<2>>({0, 0}, {dim, N_samp});
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i, const int j) {
            samples(i, j) /= std::sqrt(covar_diag_val);
        });
        TestStandardNormalSamples(samples);
        TestGaussianLogPDF(samples, samples_pdf, samples_gradpdf, dim*std::log(covar_diag_val), std::sqrt(covar_diag_val), abs_margin);
    }

    SECTION( "Diagonal Covariance, unit mean in all dimensions" ) {
        std::shared_ptr<Distribution<Kokkos::HostSpace>> dist = CreateGaussianDistribution<Kokkos::HostSpace>(mean, covar);
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> samples ("sample matrix", dim, N_samp);
        Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> samples_pdf ("sample pdf", N_samp);
        Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> samples_gradpdf ("sample grad pdf", dim, N_samp);
        dist->SetSeed(seed);
        dist->SampleImpl(samples);
        dist->LogDensityImpl(samples, samples_pdf);
        dist->LogDensityInputGradImpl(samples, samples_gradpdf);
        policy = Kokkos::MDRangePolicy<typename MemoryToExecution<Kokkos::HostSpace>::Space, Kokkos::Rank<2>>({0, 0}, {dim, N_samp});
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i, const int j) {
            samples(i, j) -= 1.0;
            samples(i, j) /= std::sqrt(covar_diag_val);
        });
        TestStandardNormalSamples(samples);
        TestGaussianLogPDF(samples, samples_pdf, samples_gradpdf, dim*std::log(covar_diag_val), std::sqrt(covar_diag_val), abs_margin);
    }
}