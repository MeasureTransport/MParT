#ifndef TEST_DISTRIBUTIONS_COMMON_H
#define TEST_DISTRIBUTIONS_COMMON_H

#include<algorithm>
#include<numeric>
#include <catch2/catch_all.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/DensityBase.h"
#include "MParT/Distributions/SampleGenerator.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/Distributions/Distribution.h"

using namespace mpart;
using namespace Catch;

// Uniform generator on [0,e]^N
// TODO: Test on GPU
template<typename MemorySpace>
class UniformSampler: public SampleGenerator<MemorySpace> {
public:
// Set a given seed for this test
UniformSampler(int dim, double scale = std::exp(1.)): SampleGenerator<MemorySpace>(dim, seed), scale_(scale) {}

void SampleImpl(StridedMatrix<double, MemorySpace> output) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>,typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {output.extent(0), output.extent(1)});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j) {
        auto rgen = this->rand_pool.get_state();
        output(i,j) = scale_*rgen.drand();
        this->rand_pool.free_state(rgen);
    });
}
private:
static const unsigned int seed = 160258;
const double scale_;
};

// Uniform density on [0,e]^2
template <typename MemorySpace>
class UniformDensity : public DensityBase<MemorySpace> {
public:
UniformDensity(int dim): DensityBase<MemorySpace>(dim) {}
void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) override {
    double euler = std::exp(1.0);
    unsigned int N = pts.extent(1);
    Kokkos::parallel_for( "uniform log density", N, KOKKOS_LAMBDA (const int& j) {
        bool in_bounds1 = (pts(0, j) >= 0.0) && (pts(0, j) <= euler);
        bool in_bounds2 = (pts(1, j) >= 0.0) && (pts(1, j) <= euler);
        output(j) = in_bounds1 && in_bounds2 ? -2 : -std::numeric_limits<double>::infinity();
    });
}

void LogDensityInputGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) override {
    unsigned int N = pts.extent(1);
    Kokkos::parallel_for( "uniform grad log density", N, KOKKOS_LAMBDA (const int& j) {
        output(0,j) = 0.;
        output(1,j) = 0.;
    });
}
};

// Tests samples that should be transformed to a standard normal distribution
void TestStandardNormalSamples(StridedMatrix<double, Kokkos::HostSpace> samples);

#endif //MPART_TEST_DISTRIBUTIONS_COMMON_H
