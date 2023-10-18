#ifndef MPART_SERIALIZATION_H
#define MPART_SERIALIZATION_H

#include <cereal/types/base_class.hpp>
#include <cereal/types/string.hpp>
#include <cereal/archives/binary.hpp>

#include <Kokkos_Core.hpp>

#include "MParT/Utilities/ArrayConversions.h"

// A macro that can be used for registering the various MonotoneComponent classes with CEREAL
// This macro is used in the MapFactoryImpl*.cpp files 
#define REGISTER_MONO_COMP(BASIS_TYPE, POS_TYPE, QUAD_TYPE, MEMORY_SPACE) CEREAL_REGISTER_TYPE(mpart::MonotoneComponent<mpart::MultivariateExpansionWorker<BASIS_TYPE, MEMORY_SPACE>, mpart::POS_TYPE, mpart::QUAD_TYPE<MEMORY_SPACE>, MEMORY_SPACE>)


namespace cereal {
    template <typename ScalarType, typename Archive, typename... Traits>
    std::enable_if_t<traits::is_output_serializable<BinaryData<ScalarType>, Archive>::value && (std::is_same_v<Traits, Kokkos::HostSpace> || ...), void> save(
        Archive &ar, Kokkos::View<ScalarType*, Traits...> const &vec) {
        Kokkos::View<ScalarType*,Traits...> vec_h = vec;
        static_assert(!(std::is_same_v<Traits, Kokkos::LayoutStride> || ...), "LayoutStride not supported");

        std::string name = vec.label();
        ar(name);

        unsigned int sz = vec_h.extent(0);
        ar(sz);

        if(sz>0){
            ar(binary_data(vec_h.data(), sz * sizeof(std::remove_cv_t<ScalarType>)));
        }
    }

    template<typename ScalarType, typename Archive, typename... Traits>
    std::enable_if_t<traits::is_input_serializable<BinaryData<ScalarType>, Archive>::value && (std::is_same_v<Traits, Kokkos::HostSpace> || ...), void> load(
        Archive &ar, Kokkos::View<ScalarType*, Traits...> &vec) {
        static_assert(!(std::is_same_v<Traits, Kokkos::LayoutStride> || ...), "LayoutStride not supported");
        
        std::string name;
        ar(name);
        unsigned int sz;
        ar(sz);
        Kokkos::View<ScalarType*,Traits...> vec_h (name, sz);

        if(sz>0){
            ar(binary_data(vec_h.data(), sz * sizeof(ScalarType)));
        }
        vec = std::move(vec_h);
    }

    template<typename ScalarType, typename Archive, typename... Traits>
    std::enable_if_t<traits::is_output_serializable<BinaryData<ScalarType>, Archive>::value && (std::is_same_v<Traits, Kokkos::HostSpace> || ...), void> save(
        Archive &ar, Kokkos::View<ScalarType**, Traits...> const &mat) {
        static_assert(!(std::is_same_v<Traits, Kokkos::LayoutStride> || ...), "LayoutStride not supported");

        std::string name = mat.label();
        ar(name);

        unsigned int m = mat.extent(0);
        unsigned int n = mat.extent(1);
        ar(m,n);

        if((m>0)&&(n>0)){
            Kokkos::View<ScalarType**,Traits...> mat_h = mat;
            ar(binary_data(mat_h.data(), m * n * sizeof(ScalarType)));
        }
    }

    template<typename ScalarType, typename Archive, typename... Traits>
    std::enable_if_t<traits::is_input_serializable<BinaryData<ScalarType>, Archive>::value && (std::is_same_v<Traits, Kokkos::HostSpace> || ...), void> load(
        Archive &ar, Kokkos::View<ScalarType**, Traits...> &mat) {
        static_assert(!(std::is_same_v<Traits, Kokkos::LayoutStride> || ...), "LayoutStride not supported");
        std::string name;
        ar(name);
        unsigned int m,n;
        ar(m,n);
        Kokkos::View<ScalarType**,Traits...> mat_h (name, m, n);
        if((m>0)&&(n>0)){
            ar(binary_data(mat_h.data(), m * n * sizeof(ScalarType)));
        }
        mat = std::move(mat_h);
    }
}

#endif // MPART_SERIALIZATION_H