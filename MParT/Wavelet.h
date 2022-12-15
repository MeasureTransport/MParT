#ifndef MPART_WAVELET_H
#define MPART_WAVELET_H

#include <Kokkos_Core.hpp>
#include <cmath>
#include "MParT/Utilities/MathFunctions.h"

namespace mpart {
// Assume wavelet has support on [a,b].
// Assume data is represented on [c,d].
// Initial scale (s_2) and shift (s_1) are to map [c,d] to [a,b].
// p_{l,q}(x;s_1,s_2) = (2^l/2)*psi(2^l*(x*s_2-s_1)-q)
// where psi is the mother wavelet function.
// Order of a wavelet is d = 2^l+q
template<class Mixer>
class Wavelet: public Mixer {

public:
    Wavelet(double c, double d): scale_((Mixer::b-Mixer::a)/(d-c)), shift_(c*(Mixer::b-Mixer::a)/(d-c)-Mixer::a) {}

    /* Evaluates all wavelets up to a specified order. */
    KOKKOS_FUNCTION void EvaluateAll(double*              output,
                                    unsigned int         maxOrder,
                                    double               x) const
    {
        x *= scale_;
        x -= shift_;
        if(x > Mixer::b || x < Mixer::a) {
            std::fill(output, output+maxOrder, 0.);
            return;
        }
        unsigned int L,Q;
        MaxOrder(maxOrder, L, Q);
        unsigned int idx = 2;
        double outscale = M_SQRT2;
        unsigned int two_power_ell = 2;
        output[0] = 0.;
        output[1] = this->psi(x);
        x *= 2;
        for(unsigned int l=1; l<=L; ++l)
        {
            unsigned int Q_l = l == L ? Q+1 : two_power_ell;

            for(unsigned int q=0; q < Q_l; ++q)
            {
                output[idx] = outscale*this->psi(x-(Mixer::a + (2*q+1)*(Mixer::b-Mixer::a)/(2*two_power_ell)));
                ++idx;
            }
            x *= 2;
            outscale *= M_SQRT2;
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
        x *= scale_;
        x -= shift_;
        if(x > Mixer::b || x < Mixer::a) {
            std::fill(vals, vals+maxDegree, 0.);
            std::fill(derivs, derivs+maxDegree, 0.);
            return;
        }
        unsigned int L,Q;
        MaxOrder(maxDegree, L, Q);
        unsigned int idx = 2;
        double outscale = M_SQRT2;
        double outscale_deriv_inc = 2*M_SQRT2;
        double outscale_deriv = scale_*outscale_deriv_inc;
        unsigned int two_power_ell = 2;
        vals[0] = 0.;
        vals[1] = this->psi(x);
        derivs[0] = 0.;
        derivs[1] = scale_*this->psi_derivative(x);
        x *= 2;
        for(unsigned int l=1; l<=L; ++l)
        {
            unsigned int Q_l = l == L ? Q+1 : two_power_ell;
            for(unsigned int q=0; q < Q_l; ++q)
            {
                vals[idx] = outscale*this->psi(x-q);
                derivs[idx] = outscale_deriv*this->psi_derivative(x-q);
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
        x *= scale_;
        x -= shift_;
        if(x > Mixer::b || x < Mixer::a) {
            std::fill(vals, vals+maxDegree, 0.);
            std::fill(derivs, derivs+maxDegree, 0.);
            std::fill(derivs2, derivs2+maxDegree, 0.);
            return;
        }
        unsigned int L,Q;
        MaxOrder(maxDegree, L, Q);
        unsigned int idx = 2;
        double outscale = M_SQRT2;
        double outscale_deriv_inc = 2*M_SQRT2;
        double outscale_deriv = scale_*outscale_deriv_inc;
        double outscale_deriv2_inc = 4*M_SQRT2;
        double outscale_deriv2 = scale_*scale_*outscale_deriv2_inc;
        unsigned int two_power_ell = 2;
        vals[0] = 0.;
        vals[1] = this->psi(x);
        derivs[0] = 0.;
        derivs[1] = scale_*this->psi_derivative(x);
        derivs2[0] = 0.;
        derivs2[1] = scale_*scale_*this->psi_second_derivative(x);
        x *= 2;
        for(unsigned int l=1; l<=L; ++l)
        {
            unsigned int Q_l = l == L ? Q+1 : two_power_ell;
            for(unsigned int q=0; q < Q_l; ++q)
            {
                vals[idx] = outscale*this->psi(x-q);
                derivs[idx] = outscale_deriv*this->psi_derivative(x-q);
                derivs2[idx] = outscale_deriv2*this->psi_second_derivative(x-q);
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
        x *= scale_;
        x -= shift_;
        if(x > Mixer::b || x < Mixer::a) {
            return 0.;
        }
        unsigned int ell,q;
        MaxOrder(order, ell, q);
        unsigned int sqrtscale = 1 << (ell/2);
        unsigned int inscale = 1 << ell;
        double val = sqrtscale*this->psi(inscale*x - (Mixer::a + (2*q+1)*(Mixer::b-Mixer::a)/(2*inscale)));
        if(ell % 2 == 1) val *= M_SQRT2;
        return val;
    }

    KOKKOS_INLINE_FUNCTION double Derivative(unsigned int order, double x) const
    {
        x *= scale_;
        x -= shift_;
        if(x > Mixer::b || x < Mixer::a) {
            return 0.;
        }
        unsigned int ell,q;
        MaxOrder(order, ell, q);
        unsigned int sqrtscale = 1 << (3*ell/2);
        unsigned int inscale = 1 << ell;
        double val = scale_*sqrtscale*this->psi_derivative(inscale*x - q);
        if(ell % 2 == 1) val *= M_SQRT2;
        return val;
    }

    KOKKOS_INLINE_FUNCTION double SecondDerivative(unsigned int order, double x) const
    {
        x *= scale_;
        x -= shift_;
        if(x > Mixer::b || x < Mixer::a) {
            return 0.;
        }
        unsigned int ell,q;
        MaxOrder(order, ell, q);
        unsigned int sqrtscale = 1 << (5*ell/2);
        unsigned int inscale = 1 << ell;
        double val = scale_*scale_*sqrtscale*this->psi_second_derivative(inscale*x - q);
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
    return normalization*exp(-x_sq_frac/2)*(1-x_sq_frac);
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
static double constexpr a = -6.;
static double constexpr b = 6.;

private:
static double constexpr sigma_sqrt = 1.;
static double constexpr sigma = sigma_sqrt*sigma_sqrt;
static double constexpr sigma_sq = sigma*sigma;
static double constexpr normalization = 8/(3*sqrt_3_v*quartroot_pi_v);
}; // namespace mpart

using RickerWavelet = Wavelet<RickerWaveletMixer>;

} // namespace mpart

#endif // MPART_WAVELET_H