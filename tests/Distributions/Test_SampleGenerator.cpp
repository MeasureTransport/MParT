#include<algorithm>
#include <catch2/catch_all.hpp>
#include <Kokkos_Random.hpp>
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/SampleGenerator.h"

using namespace mpart;
using namespace Catch;


template<class GeneratorPool>
struct uniform_gen_functor{
    using rnd_type = typename GeneratorPool::generator_type;
    GeneratorPool rand_pool;
    StridedMatrix<double, Kokkos::HostSpace> output;

    uniform_gen_functor(GeneratorPool rand_pool_, StridedMatrix<double, Kokkos::HostSpace> output_) : rand_pool(rand_pool_), output(output_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int& j) const {
        rnd_type rgen = rand_pool.get_state();
        output(0,j) = Kokkos::rand<rnd_type, double>::draw(rgen);
    }
};

// Uniform generator on [0,1]
class UniformGenerator {
public:

using PoolType = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
PoolType rand_pool;

UniformGenerator(): rand_pool() {}

void SampleImpl(Kokkos::View<double**, Kokkos::HostSpace> output) {
    unsigned int N = output.extent(1);
    Kokkos::parallel_for("uniform generator", N, KOKKOS_LAMBDA(const int& j) {
        typename PoolType::generator_type rgen = rand_pool.get_state();
        output(0,j) = rgen.drand();
    });
}
};

TEST_CASE( "Testing SampleGenerator", "[SampleGenerator]") {
// Sample 1000 points
// Check empirical CDF against uniform CDF
// assert the difference is less than something
    auto generator = std::make_shared<UniformGenerator>();
    unsigned int N_pts = 1000;
    SECTION("SampleImpl") {
        Kokkos::View<double**, Kokkos::HostSpace> output ("output", 1, N_pts);
        generator->SampleImpl(output);
        auto vec_view = Kokkos::subview(output, 0, Kokkos::ALL());
        std::vector<int> idx (N_pts);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&vec_view](int i1, int i2) {return vec_view(i1) < vec_view(i2);});
        for(unsigned int j = 0; j < N_pts; ++j) {
            double ecdf_j = ((double)idx[j])/((double)N_pts);
            REQUIRE(ecdf_j == Approx(output(0,j)/2.0).epsilon(1e-3));
        }
    }
}
