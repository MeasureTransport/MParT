#ifndef JLARRAYCONVERSIONS_H
#define JLARRAYCONVERSIONS_H

#include <Kokkos_Core.hpp>
#include <Kokkos_Layout.hpp>
#include <Eigen/Core>
#include "jlcxx/jlcxx.hpp"
#include "jlcxx/functions.hpp"


template<typename ScalarType>
inline Kokkos::View<ScalarType*, Kokkos::HostSpace> VecToKokkos(std::vector<double> const& v) {
    return mpart::ToKokkos(&arr[0], arr.size());
}

template<typename ScalarType>
inline jlcxx::ArrayRef<ScalarType> VecToJulia(Kokkos::View<ScalarType*, Kokkos::HostSpace> view) {
    return jlcxx::ArrayRef<ScalarType>(view.data(), view.extent(0));
}

#endif // JLARRAYCONVERSIONS_H