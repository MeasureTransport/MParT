#ifndef MPART_UNIVARIATEBASES_H
#define MPART_UNIVARIATEBASES_H

#include <Kokkos_Core.hpp>
#include <cmath>

#include <iostream>

#include "MParT/Utilities/MathFunctions.h"

namespace mpart{

/**
 * @brief Generic class to represent functions
 * \f[p_0(x)\equiv 1,p_k(x)=sin(2\pi k x)\f]
 */
class SineBasis
{
public:
    static constexpr double PI2 = 2*M_PI;

    /* Evaluates all polynomials up to a specified order. */
    KOKKOS_FUNCTION void EvaluateAll(double*              output,
                                     unsigned int         maxOrder,
                                     double               x) const
    {
        output[0] = 1.;
        
        double sin_x, cos_x, cos_kx;

        if(maxOrder>0) {
            sin_x = sin(PI2*x);
            cos_x = cos(PI2*x);
            cos_kx = cos_x;
            output[1] = sin_x;
        }

        for(unsigned int order=2; order<=maxOrder; ++order) {
            output[order] = output[order-1]*cos_x + cos_kx*sin_x;
            cos_kx = cos_kx*cos_x - output[order-1]*sin_x;
        }
    }

    /** Evaluates the derivative of every polynomial in this family up to degree maxOrder (inclusive).
        The results are stored in the memory pointed to by the derivs pointer.
    */
    KOKKOS_FUNCTION void EvaluateDerivatives(double*      derivs,
                             unsigned int maxOrder,
                             double       x) const
    {
        derivs[0] = 0.;
        
        double sin_x, cos_x, sin_kx;

        if(maxOrder>0) {
            sin_x = sin(PI2*x);
            cos_x = cos(PI2*x);
            sin_kx = sin_x;
            derivs[1] = cos_x;
        }

        // d_x sin(2pi k x) = 2pi k cos(2pi k x)
        for(unsigned int order=2; order<=maxOrder; ++order) {
            derivs[order] = derivs[order-1]*cos_x - sin_kx*sin_x;
            sin_kx = sin_kx*cos_x + derivs[order-1]*sin_x;
            derivs[order-2] *= PI2*(order-2);
        }
        if(maxOrder>0) derivs[maxOrder-1] *= PI2*(maxOrder-1);
        derivs[maxOrder] *= PI2*(maxOrder);
    }

    /** Evaluates the value and derivative of every polynomial in this family up to degree maxOrder (inclusive).
        The results are stored in the memory pointed to by the derivs pointer.
    */
    KOKKOS_FUNCTION void EvaluateDerivatives(double*      vals,
                           double*      derivs,
                           unsigned int maxOrder,
                           double       x) const
    {
        vals[0] = 1.;
        derivs[0] = 0.;
        
        double sin_x, cos_x;

        if(maxOrder>0) {
            sin_x = sin(PI2*x);
            cos_x = cos(PI2*x);
            vals[1] = sin_x;
            derivs[1] = cos_x;
        }

        for(unsigned int order=2; order<=maxOrder; ++order) {
            vals[order] = vals[order-1]*cos_x + derivs[order-1]*sin_x;
            derivs[order] = derivs[order-1]*cos_x - vals[order-1]*sin_x;
            derivs[order-2] *= PI2*(order-2);
        }
        if(maxOrder>0) derivs[maxOrder-1] *= PI2*(maxOrder-1);
        derivs[maxOrder] *= PI2*(maxOrder);
    }

    KOKKOS_FUNCTION void EvaluateSecondDerivatives(double*      vals,
                                   double*      derivs,
                                   double*      secondDerivs,
                                   unsigned int maxOrder,
                                   double       x) const
    {
        vals[0] = 1.;
        derivs[0] = 0.;
        secondDerivs[0] = 0.;
        
        double sin_x, cos_x;

        if(maxOrder>0) {
            sin_x = sin(PI2*x);
            cos_x = cos(PI2*x);
            vals[1] = sin_x;
            derivs[1] = cos_x;
            secondDerivs[1] = -PI2*PI2*vals[1];
        }

        for(unsigned int order=2; order<=maxOrder; ++order) {
            vals[order] = vals[order-1]*cos_x + derivs[order-1]*sin_x;
            derivs[order] = derivs[order-1]*cos_x - vals[order-1]*sin_x;
            derivs[order-2] *= PI2*(order-2);
            secondDerivs[order] = -(PI2*order)*(PI2*order)*vals[order];
        }
        if(maxOrder > 0) derivs[maxOrder-1] *= PI2*(maxOrder-1);
        derivs[maxOrder] *= PI2*(maxOrder);
    }



    KOKKOS_FUNCTION double Evaluate(unsigned int const order,
                    double const x) const
    {
        return order == 0 ? 1. : sin(PI2*order*x);
    }

    KOKKOS_FUNCTION double Derivative(unsigned int const order,
                      double const x) const
    {
        return order == 0 ? 0. : PI2*order*cos(PI2*order*x);
    }

    KOKKOS_FUNCTION double SecondDerivative(unsigned int const order,
                      double const x) const
    {
        return -(PI2*order)*(PI2*order)*sin(PI2*order*x);
    }

};

} // namespace mpart

#endif