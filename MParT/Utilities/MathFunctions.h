#ifndef MPART_MATHFUNCTIONS_H
#define MPART_MATHFUNCTIONS_H

#include <Kokkos_Core.hpp>

namespace mpart{

    /** Computes the factorial d! */
    KOKKOS_INLINE_FUNCTION unsigned int Factorial(unsigned int d)
    {
        unsigned int out = 1;
        for(unsigned int i=2; i<=d; ++i)
            out *= i;
        return out;
    }

    /** Computes log_2(x) */
    KOKKOS_INLINE_FUNCTION unsigned int Log2(unsigned int x)
    {
        unsigned int out = 0;
        while(x >>= 1) ++out;
        return out;
    }

    constexpr double quartroot_pi_v = 1.772454211850187330531040021040;
    constexpr double sqrt_3_v =       1.732050807568877293527446341506;
}

#endif