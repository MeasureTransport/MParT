#ifndef MPART_MATHFUNCTIONS_H
#define MPART_MATHFUNCTIONS_H

#include <Kokkos_Core.hpp>
#include "ArrayConversions.h"

namespace mpart{

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