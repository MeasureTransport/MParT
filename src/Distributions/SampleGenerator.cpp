#include "MParT/Distributions/SampleGenerator.h"

using namespace mpart;

template<>
Eigen::RowMatrixXd SampleGenerator<Kokkos::HostSpace>::SampleEigen(unsigned int N) {
    // Allocate output
    Eigen::RowMatrixXd out(dimension, N);
    StridedMatrix<double, Kokkos::HostSpace> outView = RowMatToKokkos<double, Kokkos::HostSpace>(out);
    // Call the Sample function
    auto output_kokkos = Sample(N, outView);

    return out;
}

template<typename MemorySpace>
StridedMatrix<double, MemorySpace> SampleGenerator<MemorySpace>::Sample(unsigned int N) {
    // Allocate output
    StridedMatrix<double, MemorySpace> out("Generated Samples", dimension, N);
    // Call the SampleImpl function
    SampleImpl(out);

    return out;
}