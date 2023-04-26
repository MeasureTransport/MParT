#include "JlArrayConversions.h"

namespace mpart {
namespace binding {
/**
 * @brief Wrap a Julia vector in a Kokkos View
 *
 * @param vec Julia vector to take view of
 * @return mpart::StridedVector<double, Kokkos::HostSpace> Same memory, but now a Kokkos View
 */
StridedVector<double, Kokkos::HostSpace> JuliaToKokkos(jlcxx::ArrayRef<double,1> &vec)
{
    double* vptr = vec.data();
    unsigned int vsize = vec.size();
    return ToKokkos(vptr, vsize);
}

/**
 * @brief Wrap a Julia matrix in a Kokkos View
 *
 * @param mat Julia matrix to take view of
 * @return mpart::StridedMatrix<double, Kokkos::HostSpace> Same memory, but now a Kokkos View
 */
StridedMatrix<double, Kokkos::HostSpace> JuliaToKokkos(jlcxx::ArrayRef<double,2> &mat)
{
    double* mptr = mat.data();
    unsigned int rows = size(mat,0);
    unsigned int cols = size(mat,1);

    return ToKokkos<double,Kokkos::LayoutLeft>(mptr, rows, cols);
}

/**
 * @brief Wrap a Julia matrix in an Eigen Map
 *
 * @param mat Reference to the Julia matrix to wrap
 * @return auto an Eigen Map to the same memory as the Julia matrix
 */
Eigen::Map<const Eigen::Matrix<int,Eigen::Dynamic, Eigen::Dynamic>,0,Eigen::OuterStride<>> JuliaToEigenMat(jlcxx::ArrayRef<int,2> mat) {
    int* mptr = mat.data();
    unsigned int rows = size(mat,0);
    return Eigen::Map<const Eigen::Matrix<int,Eigen::Dynamic, Eigen::Dynamic>,0,Eigen::OuterStride<>>(mptr, rows, cols, Eigen::OuterStride<>(rows));
}

/**
 * @brief Wrap a 1D Kokkos View in a Julia vector
 *
 * @param view View to access in Julia
 * @return jlcxx::ArrayRef<double> Same memory, but now a Julia array
 */
jlcxx::ArrayRef<double> KokkosToJulia(StridedVector<double, Kokkos::HostSpace> view) {
    double* vptr = view.data();
    unsigned int sz = view.extent(0);
    return jlcxx::make_julia_array(vptr, sz);
}

/**
 * @brief Wrap a 2D Kokkos View in a Julia matrix
 *
 * @param view View to access in Julia
 * @return jlcxx::ArrayRef<double,2> Same memory, but now a Julia array
 */
jlcxx::ArrayRef<double,2> KokkosToJulia(StridedMatrix<double, Kokkos::HostSpace> view) {
    double* vptr = view.data();
    unsigned int rows = view.extent(0);
    unsigned int cols = view.extent(1);
    return jlcxx::make_julia_array(vptr, rows, cols);
}
} // namespace binding
} // namespace mpart