#ifndef MPART_WAVELET_H
#define MPART_WAVELET_H

#include <Kokkos_Core.hpp>
#include <cmath>

#include "MParT/Utilities/MathFunctions.h"

namespace mpart {
// Assume wavelet has support on [a,b].
// Assume data is represented on [c,d].
// Initial shift s_1 = c-a
// Initial scale s_2 = (d-c)/(b-a)
// p_{l,q}(x;s_1,s_2) = (2^l/2)*psi(2^l*(x-s_1)*s_2-q)
// where psi is the mother wavelet function.
template<class Mixer>
class Wavelet: public Mixer {

public:
    Wavelet(unsigned int c, unsigned int d): shift_(c-Mixer::a), scale_((d-c)/(Mixer::b-Mixer::a)) {}

    /* Evaluates all wavelets up to a specified order. */
    KOKKOS_FUNCTION void EvaluateAll(double*              output,
                                    unsigned int         maxOrder,
                                    double               x) const
    {
        unsigned int L,Q;
        MaxOrder(maxOrder, L, Q);
        unsigned int idx = 1;
        double outscale = M_SQRT2;
        unsigned int two_power_ell = 2;
        x -= shift_;
        x *= scale_;
        output[0] = psi(x);
        x *= 2;
        for(unsigned int l=1; l<=L; ++l)
        {
            unsigned int Q_l = l == L ? Q+1 : two_power_ell;
            for(unsigned int q=0; q < Q_l; ++q)
            {
                output[idx] = outscale*psi(x-q);
                ++idx;
            }
            x *= 2;
            outscale *= 2;
            two_power_ell *= 2;
        }
    }


    /** Evaluates the derivative of every wavelet in this family up to degree maxOrder (inclusive).
        The results are stored in the memory pointed to by the derivs pointer.
    */
    KOKKOS_FUNCTION void EvaluateDerivatives(double* vals,
                             double*      derivs,
                             unsigned int maxDegree,
                             double       x) const
    {
        unsigned int L,Q;
        MaxOrder(maxOrder, L, Q);
        unsigned int idx = 1;
        double outscale = M_SQRT2;
        double outscale_deriv_inc = 2*M_SQRT2;
        double outscale_deriv = scale_*outscale_deriv_inc;
        unsigned int two_power_ell = 2;
        x -= shift_;
        x *= scale_;
        vals[0] = psi(x);
        derivs[0] = scale_*psi_derivative(x);
        x *= 2;
        for(unsigned int l=1; l<=L; ++l)
        {
            unsigned int Q_l = l == L ? Q+1 : two_power_ell;
            for(unsigned int q=0; q < Q_l; ++q)
            {
                vals[idx] = outscale*psi(x-q);
                derivs[idx] = outscale_deriv*psi_derivative(x-q);
                ++idx;
            }
            x *= 2;
            outscale_deriv *= outscale_deriv_inc;
            two_power_ell *= 2;
        }
    }

    /** Evaluates the second derivative of every wavelet in this family up to degree maxOrder (inclusive).
        The results are stored in the memory pointed to by the derivs pointer.
    */
    KOKKOS_FUNCTION void EvaluateSecondDerivatives(double* vals,
                             double*      derivs,
                             double*      derivs2,
                             unsigned int maxDegree,
                             double       x) const
    {
        unsigned int L,Q;
        MaxOrder(maxOrder, L, Q);
        unsigned int idx = 1;
        double outscale = M_SQRT2;
        double outscale_deriv_inc = 2*M_SQRT2;
        double outscale_deriv = scale_*outscale_deriv_inc;
        double outscale_deriv2_inc = 4*M_SQRT2;
        double outscale_deriv2 = scale_*scale_*outscale_deriv2_inc;
        unsigned int two_power_ell = 2;
        x -= shift_;
        x *= scale_;
        vals[0] = psi(x);
        derivs[0] = scale_*psi_derivative(x);
        derivs2[0] = scale_*scale_*psi_second_derivative(x);
        x *= 2;
        for(unsigned int l=1; l<=L; ++l)
        {
            unsigned int Q_l = l == L ? Q+1 : two_power_ell;
            for(unsigned int q=0; q < Q_l; ++q)
            {
                vals[idx] = outscale*psi(x-q);
                derivs[idx] = outscale_deriv*psi_derivative(x-q);
                derivs2[idx] = outscale_deriv2*psi_second_derivative(x-q);
                ++idx;
            }
            x *= 2;
            outscale_deriv *= outscale_deriv_inc;
            outscale_deriv2 *= outscale_deriv2_inc;
            two_power_ell *= 2;
        }
    }

    KOKKOS_INLINE_FUNCTION double Evaluate(unsigned int order, double x) const
    {
        unsigned int ell,q;
        MaxOrder(order, ell, q);
        unsigned int sqrtscale = 1 << (ell/2);
        unsigned int inscale = 1 << ell;
        x -= shift_;
        x *= scale_;
        double val = sqrtscale*psi(inscale*x - q);
        if(ell % 2 == 1) val *= M_SQRT2;
        return val;
    }

    KOKKOS_INLINE_FUNCTION double Derivative(unsigned int order, double x) const
    {
        unsigned int ell,q;
        MaxOrder(order, ell, q);
        unsigned int sqrtscale = 1 << (3*ell/2);
        unsigned int inscale = 1 << ell;
        x -= shift_;
        x *= scale_;
        double val = scale_*sqrtscale*psi_derivative(inscale*x - q);
        if(ell % 2 == 1) val *= M_SQRT2;
        return val;
    }

    KOKKOS_INLINE_FUNCTION double SecondDerivative(unsigned int order, double x) const
    {
        unsigned int ell,q;
        MaxOrder(order, ell, q);
        unsigned int sqrtscale = 1 << (5*ell/2);
        unsigned int inscale = 1 << ell;
        x -= shift_;
        x *= scale_;
        double val = scale_*scale_*sqrtscale*psi_second_derivative(inscale*x - q);
        if(ell % 2 == 1) val *= M_SQRT2;
        return val;
    }
private:
    KOKKOS_INLINE_FUNCTION void MaxOrder(unsigned int order, unsigned int& L, unsigned int& Q) const
    {
        L = Log2(order);
        Q = order - L;
    }
    double shift_, scale_;
};

class RickerWaveletMixer {
protected:
KOKKOS_INLINE_FUNCTION double psi(double x) const
{
    double x_sq_frac = (x*x)/sigma_sq;
    return normalization*exp(-x_sq_frac/2)*(1-2*x_sq_frac);
}
KOKKOS_INLINE_FUNCTION double psi_derivative(double x) const
{
    double x_sq_frac = (x*x)/sigma_sq;
    return normalization*exp(-x_sq_frac/2)*x*(x_sq_frac-3)/sigma_sq;
}
KOKKOS_INLINE_FUNCTION double psi_second_derivative(double x) const
{
    double x_sq_frac = (x*x)/sigma_sq;
    return normalization*exp(-x_sq_frac/2)*(3 - 6*x_sq_frac/sigma_sq + x_sq_frac*x_sq_frac)/sigma_sq;
}
static double const a = -6.;
static double const b = 6.;

private:
static double const sigma = 1.;
static double const sigma_sq = sigma*sigma;
static double const normalization = 2/sqrt(3*sigma*sqrt(M_PI));
} // namespace mpart

#endif // MPART_WAVELET_H