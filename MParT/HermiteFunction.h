#ifndef MPART_HERMITEFUNCTION_H
#define MPART_HERMITEFUNCTION_H

#include "MParT/OrthogonalPolynomial.h"

namespace mpart{

class HermiteFunction 
{
public:

    void EvaluateAll(double*              output,
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

    double Evaluate(unsigned int const order, 
                    double const x, 
                    unsigned int currDim = 0) const
    {
        const double scaling = std::pow( std::pow(2, order) *
                               std::tgamma(order+1) *
                               std::sqrt(M_PI), -0.5) * std::exp(-0.5*x*x);

        return scaling * polyBase.Evaluate(order, x);

    }

private:
    PhysicistHermite polyBase;

}; // class HermiteFunction

}

#endif 