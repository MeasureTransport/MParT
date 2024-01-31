#ifndef MPART_MATHFUNCTIONS_H
#define MPART_MATHFUNCTIONS_H

#include <Kokkos_Core.hpp>
#include "ArrayConversions.h"

namespace mpart{

    #if (KOKKOS_VERSION / 10000 == 3) && (KOKKOS_VERSION / 100 % 100 < 7)
    namespace MathSpace = Kokkos::Experimental;
    #else
    namespace MathSpace = Kokkos;
    #endif

    /** Computes the factorial d! */
    KOKKOS_INLINE_FUNCTION unsigned int Factorial(unsigned int d)
    {
        unsigned int out = 1;
        for(unsigned int i=2; i<=d; ++i)
            out *= i;
        return out;
    }

}

#endif