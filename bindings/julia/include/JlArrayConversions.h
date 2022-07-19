#ifndef JLARRAYCONVERSIONS_H
#define JLARRAYCONVERSIONS_H

#include <numeric>
#include <cstdarg>
#include <Kokkos_Core.hpp>
#include <Kokkos_Layout.hpp>
#include <Eigen/Core>
#include "MParT/Utilities/ArrayConversions.h"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"

/**
 * @brief Retrieve the size of an N-dimensional Julia array.
 *
 * @tparam ScalarType The type of the elements in the array.
 * @tparam N The dimension of the array (e.g. 1<--> vector, 2<--> matrix, etc.).
 * @param arr Array to take size of
 * @param j Dimension to take size of
 * @return unsigned int size along that dimension
 */
template<typename ScalarType, int N>
unsigned int size(jlcxx::ArrayRef<ScalarType, N> arr, unsigned int j) {
    return jl_array_size((jl_value_t*) arr.wrapped(), j);
}

/**
 * @brief Wrap a Julia vector in a Kokkos View
 *
 * @param vec Julia vector to take view of
 * @return mpart::StridedVector<double, Kokkos::HostSpace> Same memory, but now a Kokkos View
 */
mpart::StridedVector<double, Kokkos::HostSpace> JuliaToKokkos(jlcxx::ArrayRef<double,1> &vec)
{
    double* vptr = vec.data();
    unsigned int vsize = vec.size();
    return mpart::ToKokkos(vptr, vsize);
}

/**
 * @brief Wrap a Julia matrix in a Kokkos View
 *
 * @param mat Julia matrix to take view of
 * @return mpart::StridedMatrix<double, Kokkos::HostSpace> Same memory, but now a Kokkos View
 */
mpart::StridedMatrix<double, Kokkos::HostSpace> JuliaToKokkos(jlcxx::ArrayRef<double,2> &mat)
{
    double* mptr = mat.data();
    unsigned int rows = size(mat,0);
    unsigned int cols = size(mat,1);

    return mpart::ToKokkos<double,Kokkos::LayoutLeft>(mptr, rows, cols);
}

/**
 * @brief Wrap a Julia matrix in an Eigen Map
 *
 * @tparam ScalarType type of elements in the matrix
 * @param mat Reference to the Julia matrix to wrap
 * @return auto an Eigen Map to the same memory as the Julia matrix
 */
template<typename ScalarType>
auto JuliaToEigen(jlcxx::ArrayRef<ScalarType,2> mat) {
    ScalarType* mptr = mat.data();
    unsigned int rows = size(mat,0);
    unsigned int cols = size(mat,1);
    return Eigen::Map<const Eigen::Matrix<ScalarType,Eigen::Dynamic,1>,0,Eigen::OuterStride<>>(mptr, rows, cols, Eigen::OuterStride<>(std::max(rows,cols)));
}
/**
 * @brief Allocate array that Julia is responsible for
 *
 * @tparam T type of elements in the array
 * @tparam Sizes types of sizes along each dimension
 * @param dims sizes along each dimension
 * @return jlcxx::ArrayRef<T, sizeof...(Sizes)> A new Julia-managed array of size dims[0]x...xdims[sizeof...(Sizes)-1]
 */
template<typename T, typename... Sizes>
jlcxx::ArrayRef<T, sizeof...(Sizes)> jlMalloc(Sizes... dims) {

    unsigned int sz = (dims * ...);

    T* newMemory = (T*) malloc(sizeof(T)*sz);
    jlcxx::ArrayRef<T,sizeof...(Sizes)> output(true, newMemory, dims...);
    return output;
}

/**
 * @brief Wrap a 1D Kokkos View in a Julia vector
 *
 * @param view View to access in Julia
 * @return jlcxx::ArrayRef<double> Same memory, but now a Julia array
 */
jlcxx::ArrayRef<double> KokkosToJulia(mpart::StridedVector<double, Kokkos::HostSpace> view) {
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
jlcxx::ArrayRef<double,2> KokkosToJulia(mpart::StridedMatrix<double, Kokkos::HostSpace> view) {
    double* vptr = view.data();
    unsigned int rows = view.extent(0);
    unsigned int cols = view.extent(1);
    return jlcxx::make_julia_array(vptr, rows, cols);
}

#endif // JLARRAYCONVERSIONS_H