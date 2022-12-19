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

    // Computes the quantile function for the standard normal
    // distribution up to values of 1e7. Uses AS241 from Wichura, 1987
    KOKKOS_INLINE_FUNCTION double NormalQuantile(double P) {
        const double SPLIT1 = 0.425E0;
        const double SPLIT2 = 5.0E0;
        const double CONST1 = 0.180625E0;
        const double CONST2 = 1.6E0;

        const double A0 = 3.3871327179E0;
        const double A1 = 5.0434271938E1;
        const double A2 = 1.5929113202E2;
        const double A3 = 5.9109374720E1;
        const double B1 = 1.7895169469E1;
        const double B2 = 7.8757757664E1;
        const double B3 = 6.7187563600E1;

        const double C0 = 1.4234372777E0;
        const double C1 = 2.7568153900E0;
        const double C2 = 1.3067284816E0;
        const double C3 = 1.7023821103E-1;
        const double D1 = 7.3700164250E-1;
        const double D2 = 1.2021132975E-1;

        const double E0 = 6.6579051150E0;
        const double E1 = 3.0812263860E0;
        const double E2 = 4.2868294337E-1;
        const double E3 = 1.7337203997E-2;
        const double F1 = 2.4197894225E-1;
        const double F2 = 1.2258202635E-2;

        double Q = P - 0.5;
        double R;
        double PPND7;

        if (std::abs(Q) < SPLIT1) {
            R = CONST1 - Q * Q;
            PPND7 = Q * (((A3 * R + A2) * R + A1) * R + A0) /
                        (((B3 * R + B2) * R + B1) * R + 1.0);
        }
        R = (Q<0) ? P : 1.0 - P;

        if (R < 0) {
            PPND7 = std::nan("0");
        }
        R = std::sqrt(-std::log(R));
        if (R < SPLIT2) {
            R = R - CONST2;
            PPND7 = (((C3 * R + C2) * R + C1) * R + C0) /
                    ((D2 * R + D1) * R + 1.);
        }
        else {
            R = R - SPLIT2;
            PPND7 = (((E3 * R + E2) * R + E1) * R + E0) /
                    ((F2 * R + F1) * R + 1.);
        }
        if (Q < 0.) PPND7 = -PPND7;
        return PPND7;
    }

    constexpr double quartroot_pi_v = 1.772454211850187330531040021040;
    constexpr double sqrt_3_v =       1.732050807568877293527446341506;
}

#endif