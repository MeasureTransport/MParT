#include "Test_Distributions_Common.h"
using namespace Catch::Matchers;

void TestStandardNormalSamples(StridedMatrix<double, Kokkos::HostSpace> samples) {
    unsigned int dim = samples.extent(0);
    unsigned int N_samp = samples.extent(1);
    double mc_margin = (1/std::sqrt(N_samp))*4.0;

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
        CHECK_THAT(mean(i), WithinAbs(0.0, mc_margin));
        for(int j = 0; j < dim; j++) {
            double diag = (double) (i == j);
            double cov_ij = covar(i, j) - mean(i)*mean(j);
            if(i == j)
                CHECK_THAT(cov_ij, WithinRel(1., mc_margin));
            else
                CHECK_THAT(cov_ij, WithinAbs(0., mc_margin));
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
        CHECK_THAT(in_one_std[i]/(double)N_samp, WithinRel(emp_one_std, mc_margin));
        CHECK_THAT(in_two_std[i]/(double)N_samp, WithinRel(emp_two_std, mc_margin));
        CHECK_THAT(in_three_std[i]/(double)N_samp, WithinRel(emp_three_std, mc_margin));
    }
}