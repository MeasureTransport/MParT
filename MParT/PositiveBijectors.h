#ifndef MPART_POSITIVEBIJECTORS_H
#define MPART_POSITIVEBIJECTORS_H

#include <math.h>

namespace mpart{

/**
 * @brief Defines the softplus function \f$g(x) = \log(1+\exp(x))\f$.
 */
class SoftPlus{
public:

    static double Evaluate(double x){
        return std::log(1.0 + std::exp(x));
    }

    static double Derivative(double x){
        return std::exp(x) / (std::exp(x) + 1.0);
    }

    static double SecondDerivative(double x){
        return std::exp(x) / std::pow(std::exp(x) + 1.0, 2.0);
    }

    static double Inverse(double x){
        return std::log(std::exp(x) - 1.0);
    }
};

} // namespace mpart

#endif 