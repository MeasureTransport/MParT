#ifndef MPART_SampleGenerator_H
#define MPART_SampleGenerator_H

#include "Kokkos_Random.hpp"

#include "MParT/Utilities/EigenTypes.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/ArrayConversions.h"

namespace mpart {

template<typename MemorySpace>
class SampleGenerator {
    public:
    
    SampleGenerator(unsigned int dim) : dim_(dim), rand_pool() {};

    virtual ~SampleGenerator() = default;

    /**
     * @brief Generates a sample from the generator.
     * @param output The matrix where we want to store the samples.
     */
    virtual void SampleImpl(StridedMatrix<double, MemorySpace> output);

    /**
     * @brief Generates a samples from the generator.
     *
     * @param N The number of Samples from this generator
     * @return StridedMatrix<double, MemorySpace> The samples from this generator
     */
    StridedMatrix<double, MemorySpace> Sample(unsigned int N) {
        StridedMatrix<double, MemorySpace> output(N, dim_);
        SampleImpl(output);
        return output;
    };

    void SetSeed(unsigned int seed) {
        rand_pool = Kokkos::Random_XorShift64_Pool<MemorySpace>(seed);
    }

    /** Sample function with conversion from Kokkos to Eigen (and possibly copy to/from device). */
    // Eigen::RowMatrixXd SampleEigen(unsigned int N);
    const unsigned int dim_;
    using PoolType = typename Kokkos::Random_XorShift64_Pool<typename MemoryToExecution<MemorySpace>::Space>;
    
    private:
    PoolType rand_pool;
};

} // namespace mpart
#endif //MPART_SampleGenerator_H