#ifndef JLARRAYCONVERSIONS_H
#define JLARRAYCONVERSIONS_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Layout.hpp>
#include <Eigen/Core>
#include "MParT/Utilities/ArrayConversions.h"
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"


mpart::StridedVector<double, Kokkos::HostSpace> JuliaToKokkos(jlcxx::ArrayRef<double,1> &vec)
{
    double* vptr = vec.data();
    unsigned int vsize = vec.size();
    return mpart::ToKokkos(vptr, vsize);
}

mpart::StridedMatrix<double, Kokkos::HostSpace> JuliaToKokkos(jlcxx::ArrayRef<double,2> &mat)
{
    double* mptr = mat.data();
    unsigned int rows = jl_array_size((jl_value_t*)mat.wrapped(),0);
    unsigned int cols = jl_array_size((jl_value_t*)mat.wrapped(),1);

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
    unsigned int rows = jl_array_size((jl_value_t*)mat.wrapped(),0);
    unsigned int cols = jl_array_size((jl_value_t*)mat.wrapped(),1);
    return Eigen::Map<const Eigen::Matrix<int,Eigen::Dynamic,1>,0,Eigen::OuterStride<>>(mptr, rows, cols, Eigen::OuterStride<>(std::max(rows,cols)));
}

#endif // JLARRAYCONVERSIONS_H