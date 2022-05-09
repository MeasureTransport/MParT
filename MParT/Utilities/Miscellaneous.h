#ifndef MPART_UTILITIES_MISCELLANEOUS_H
#define MPART_UTILITIES_MISCELLANEOUS_H

#include <Kokkos_Core.hpp>

namespace mpart{

template<typename T> 
KOKKOS_INLINE_FUNCTION void swap(T& t1, T& t2) {
    T temp = t1;
    t1 = t2;
    t2 = temp;
}

} // namespace mpart


#endif 