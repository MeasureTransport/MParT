#ifndef MPART_HERMITEFUNCTION_H
#define MPART_HERMITEFUNCTION_H

#include <Kokkos_Core.hpp>

#include "MParT/OrthogonalPolynomial.h"

namespace mpart{

class HermiteFunction 
{
public:

    KOKKOS_INLINE_FUNCTION void EvaluateAll(double*              output,
                     unsigned int         maxOrder,
                     double               x) const
    {   
        // Evaluate all of the physicist hermite polynomials
        polyBase.EvaluateAll(output, maxOrder, x);

        // Add the scaling 
        const double baseScaling = std::pow(M_PI, -0.25) * std::exp(-0.5*x*x);
        for(unsigned int i=0; i<=maxOrder; ++i)
            output[i] *= (baseScaling * std::pow( std::pow(2, i) * std::tgamma(i+1), -0.5) );
    }

   KOKKOS_INLINE_FUNCTION  void EvaluateDerivatives(double*              vals,
                             double*              derivs,
                             unsigned int         maxOrder,
                             double               x) const
    {   
        // Evaluate all of the physicist hermite polynomials
        polyBase.EvaluateDerivatives(vals, derivs, maxOrder, x);

        // Add the scaling 
        const double baseScaling = std::pow(M_PI, -0.25) * std::exp(-0.5*x*x);
        double scale;
        for(unsigned int i=0; i<=maxOrder; ++i){
            scale = baseScaling * std::pow( std::pow(2, i) * std::tgamma(i+1), -0.5);
            derivs[i] -= x*vals[i];
            derivs[i] *= scale;
            vals[i] *= scale;
        }
    }


    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(double*              vals,
                                   double*              derivs,
                                   double*              derivs2,
                                   unsigned int         maxOrder,
                                   double               x) const
    {   
        // Evaluate all of the physicist hermite polynomials
        EvaluateDerivatives(vals, derivs, maxOrder, x);

        // Add the scaling 
        for(unsigned int i=0; i<=maxOrder; ++i)
            derivs2[i] = -(2.0*i + 1.0 - x*x)*vals[i];
    }


    KOKKOS_INLINE_FUNCTION double Evaluate(unsigned int const order, 
                    double const x) const
    {
        const double scaling = std::pow( std::pow(2, order) * std::tgamma(order+1) *std::sqrt(M_PI), -0.5);

        return scaling * std::exp(-0.5*x*x) * polyBase.Evaluate(order, x);

    }

    KOKKOS_INLINE_FUNCTION double Derivative(unsigned int const order, 
                      double const x) const 
    {
        const double scaling = std::pow( std::pow(2, order) * std::tgamma(order+1) *std::sqrt(M_PI), -0.5);
        const double expPart = std::exp(-0.5*x*x);
        return scaling * ( -x*expPart*polyBase.Evaluate(order, x) + expPart * polyBase.Derivative(order,x) );
    }

    KOKKOS_INLINE_FUNCTION double SecondDerivative(unsigned int const order, 
                            double const x) const
    {
        return -(2.0*order+1.0-x*x)*Evaluate(order, x);
    }

private:
    PhysicistHermite polyBase;

}; // class HermiteFunction

}

#endif 