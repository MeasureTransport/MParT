#ifndef MPART_SERIALIZATION_H
#define MPART_SERIALIZATION_H

#include <cereal/types/base_class.hpp>
#include <cereal/archives/binary.hpp>
#include <Kokkos_Core.hpp>

#include "MParT/Utilities/ArrayConversions.h"

namespace cereal {
    template <class Archive, class ScalarType, class MemorySpace>
    std::enable_if_t<traits::is_output_serializable<BinaryData<ScalarType>, Archive>::value, void> save(
        Archive &ar, mpart::StridedVector<ScalarType, MemorySpace> const &vec) {

        mpart::StridedVector<ScalarType,Kokkos::HostSpace> vec_h = vec;
        unsigned int sz = vec_h.extent(0);
        ar(sz);
        ar(binary_data(vec_h.data(), sz * sizeof(std::remove_cv_t<ScalarType>)));
    }

    template<class Archive, class ScalarType, class MemorySpace>
    std::enable_if_t<traits::is_input_serializable<BinaryData<ScalarType>, Archive>::value, void> load(
        Archive &ar, mpart::StridedVector<ScalarType, MemorySpace> &vec) {

        unsigned int sz;
        ar(sz);
        Kokkos::View<ScalarType*,Kokkos::HostSpace> vec_h ("vec_h", sz);
        ar(binary_data(vec_h.data(), sz * sizeof(ScalarType)));
        if(std::is_same<MemorySpace,Kokkos::HostSpace>::value) {
            vec = std::move(vec_h);
        } else {
            throw std::runtime_error("Cannot deserialize to device memory");
            // vec = std::move(mpart::ToDevice(vec_h));
        }
    }

    template <class Archive, class ScalarType, class MemorySpace>
    std::enable_if_t<traits::is_output_serializable<BinaryData<ScalarType>, Archive>::value, void> save(
        Archive &ar, mpart::StridedMatrix<ScalarType, MemorySpace> const &mat) {

        unsigned int m = mat.extent(0);
        unsigned int n = mat.extent(1);
        ar(m,n);
        mpart::StridedMatrix<ScalarType,Kokkos::HostSpace> mat_h = mat;
        ar(binary_data(mat_h.data(), m * n * sizeof(ScalarType)));
    }

    template<class Archive, class ScalarType, class MemorySpace>
    std::enable_if_t<traits::is_input_serializable<BinaryData<ScalarType>, Archive>::value, void> load(
        Archive &ar, mpart::StridedMatrix<ScalarType, MemorySpace> &mat) {

        unsigned int m,n;
        ar(m,n);
        mpart::StridedMatrix<ScalarType,Kokkos::HostSpace> mat_h ("mat_h", m, n);
        ar(binary_data(mat_h.data(), m * n * sizeof(ScalarType)));
        if(std::is_same<MemorySpace,Kokkos::HostSpace>::value) {
            mat = std::move(mat_h);
        } else {
            throw std::runtime_error("Cannot deserialize to device memory");
            // mat = std::move(mpart::ToDevice(mat_h));
        }
    }
}

#endif // MPART_SERIALIZATION_H