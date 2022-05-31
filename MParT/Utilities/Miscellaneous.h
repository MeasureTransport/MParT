#ifndef MPART_UTILITIES_MISCELLANEOUS_H
#define MPART_UTILITIES_MISCELLANEOUS_H

#include <Kokkos_Core.hpp>
#include <unordered_map>
#include <string>

namespace mpart{

    template<typename T>
    KOKKOS_INLINE_FUNCTION void swap(T& t1, T& t2) {
        T temp = std::move(t1);
        t1 = std::move(t2);
        t2 = std::move(temp);
    }

    /** Tries to read an options from a std::map.  If the key does not exist, the specified default value is returned. */
    std::string GetOption(std::unordered_map<std::string,std::string> const& map,
                          std::string                                 const& key,
                          std::string                                 const& defaultValue);
}

#endif