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
        
        output[0] = 1.0;

        if(maxOrder>0){
            output[1] = x;
            
            if(maxOrder>1){
                // Evaluate all of the physicist hermite polynomials
                output[2] = std::pow(M_PI, -0.25) * std::exp(-0.5*x*x);

                if(maxOrder>2){
                    output[3] = std::sqrt(2.0) * x * output[0];
                    for(unsigned int i=2; i<=maxOrder-2; ++i)
                        output[i+2] = (x*output[i+1]  - std::sqrt(0.5*(i-1))*output[i])/std::sqrt(0.5*i);
                }
            }
        }
    }

   KOKKOS_INLINE_FUNCTION  void EvaluateDerivatives(double*              vals,
                                                    double*              derivs,
                                                    unsigned int         maxOrder,
                                                    double               x) const
    {   
        vals[0] = 1.0;
        derivs[0] = 0.0;

        if(maxOrder>0){
            vals[0] = x;
            derivs[1] = 1.0;

            if(maxOrder>1){
                // Evaluate all of the physicist hermite polynomials
                polyBase.EvaluateDerivatives(&vals[2], &derivs[2], maxOrder-2, x);

                // Add the scaling 
                const double baseScaling = std::pow(M_PI, -0.25) * std::exp(-0.5*x*x);
                double scale;
                double currFactorial = 1;

                scale = baseScaling;
                derivs[2] -= x*vals[2];
                derivs[2] *= scale;
                vals[2] *= scale;

                for(unsigned int i=1; i<=maxOrder-2; ++i){
                    currFactorial *= i;
                    scale = baseScaling * std::pow( std::pow(2, i) * currFactorial, -0.5);
                    derivs[i+2] -= x*vals[i+2];
                    derivs[i+2] *= scale;
                    vals[i+2] *= scale;
                }
            }
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

        derivs2[0] = 0.0;
        
        if(maxOrder>0){
            derivs2[1] = 0.0;

            if(maxOrder>1){
        
                // Add the scaling 
                for(unsigned int i=0; i<=maxOrder-2; ++i)
                    derivs2[i+2] = -(2.0*i + 1.0 - x*x)*vals[i+2];
            }
        }
    }


    KOKKOS_INLINE_FUNCTION double Evaluate(unsigned int const order, 
                                           double const x) const
    {
        if(order==0){
            return 1.0;
        }else if(order==1){
            return x;
        }else{
            const double scaling = std::pow( std::pow(2, order-2) * Factorial(order-2) *std::sqrt(M_PI), -0.5);
            return scaling * std::exp(-0.5*x*x) * polyBase.Evaluate(order-2, x);
        }
    }

    KOKKOS_INLINE_FUNCTION double Derivative(unsigned int const order, 
                                             double const x) const 
    {   
        if(order==0){
            return 0.0;
        }else if(order==1){
            return 1.0;
        }else{
            const double scaling = std::pow( std::pow(2, order-2) * Factorial(order-2) *std::sqrt(M_PI), -0.5);
            const double expPart = std::exp(-0.5*x*x);
            return scaling * ( -x*expPart*polyBase.Evaluate(order-2, x) + expPart * polyBase.Derivative(order-2,x) );
        }
    }

    KOKKOS_INLINE_FUNCTION double SecondDerivative(unsigned int const order, 
                            double const x) const
    {   
        if(order<2){
            return 0; 
        }else{
            return -(2.0*order+1.0-x*x)*Evaluate(order-2, x);
        }
    }

private:
    PhysicistHermite polyBase;

    KOKKOS_INLINE_FUNCTION unsigned int Factorial(unsigned int d) const{
        unsigned int out = 1;
        for(unsigned int i=2; i<=d; ++i)
            out *= i;
        return out;
    }
    
}; // class HermiteFunction

}

#endif 