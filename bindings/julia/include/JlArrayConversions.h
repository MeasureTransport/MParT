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


template<typename ScalarType, int N>
unsigned int size(jlcxx::ArrayRef<ScalarType, N> arr, unsigned int j) {
    return jl_array_size((jl_value_t*) arr.wrapped(), j);
}

mpart::StridedVector<double, Kokkos::HostSpace> JuliaToKokkos(jlcxx::ArrayRef<double,1> &vec)
{
    double* vptr = vec.data();
    unsigned int vsize = vec.size();
    return mpart::ToKokkos(vptr, vsize);
}

mpart::StridedMatrix<double, Kokkos::HostSpace> JuliaToKokkos(jlcxx::ArrayRef<double,2> &mat)
{
    double* mptr = mat.data();
    unsigned int rows = size(mat,0);
    unsigned int cols = size(mat,1);

    return mpart::ToKokkos<double,Kokkos::LayoutLeft>(mptr, rows,cols);
}

jlcxx::ArrayRef<double> KokkosToJulia(mpart::StridedVector<double, Kokkos::HostSpace> view) {
    double* vptr = view.data();
    unsigned int sz = view.extent(0);
    return jlcxx::make_julia_array(vptr, sz);
}

jlcxx::ArrayRef<double,2> KokkosToJulia(mpart::StridedMatrix<double, Kokkos::HostSpace> view) {
    double* vptr = view.data();
    unsigned int rows = view.extent(0);
    unsigned int cols = view.extent(1);
    return jlcxx::make_julia_array(vptr, rows, cols);
}

template<typename ScalarType>
auto JuliaToEigen(jlcxx::ArrayRef<ScalarType,2> mat) {
    ScalarType* mptr = mat.data();
    unsigned int rows = size(mat,0);
    unsigned int cols = size(mat,1);
    return Eigen::Map<const Eigen::Matrix<int,Eigen::Dynamic,1>,0,Eigen::OuterStride<>>(mptr, rows, cols, Eigen::OuterStride<>(std::max(rows,cols)));
}

template<typename T, typename... Sizes>
jlcxx::ArrayRef<T, sizeof...(Sizes)> jlMalloc(Sizes... dims) {

    unsigned int sz = (dims * ...);

    T* newMemory = (T*) malloc(sizeof(T)*sz);
    jlcxx::ArrayRef<T,sizeof...(Sizes)> output(true, newMemory, dims...);
    return output;
}

#endif // JLARRAYCONVERSIONS_H