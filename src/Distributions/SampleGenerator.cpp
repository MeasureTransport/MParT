#include "MParT/Distributions/SampleGenerator.h"

using namespace mpart;

// template<>
// Eigen::RowMatrixXd SampleGenerator<Kokkos::HostSpace>::SampleEigen(unsigned int N) {
//     // Allocate output
//     Eigen::RowMatrixXd out(dimension, N);
//     StridedMatrix<double, Kokkos::HostSpace> outView = RowMatToKokkos<double, Kokkos::HostSpace>(out);
//     // Call the Sample function
//     Sample(N, outView);

//     return out;
// }