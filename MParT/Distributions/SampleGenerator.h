#ifndef MPART_SampleGenerator_H
#define MPART_SampleGenerator_H

#include "MParT/Utilities/EigenTypes.h"
#include "MParT/Utilities/ArrayConversions.h"

namespace mpart {

template<typename MemorySpace>
class SampleGenerator {
    public:
    const unsigned int dimension;
    SampleGenerator(unsigned int dim) : dimension(dim) {};

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
    StridedMatrix<double, MemorySpace> Sample(unsigned int N);

    /** Sample function with conversion from Kokkos to Eigen (and possibly copy to/from device). */
    Eigen::RowMatrixXd SampleEigen(unsigned int N);
}

} // namespace mpart
#endif //MPART_SampleGenerator_H