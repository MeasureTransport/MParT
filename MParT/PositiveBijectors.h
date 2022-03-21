#ifndef MPART_POSITIVEBIJECTORS_H
#define MPART_POSITIVEBIJECTORS_H

#include <cmath>

namespace mpart{

/**
 * @brief Defines the softplus function \f$g(x) = \log(1+\exp(x))\f$.
 */
class SoftPlus{
public:

    static double Evaluate(double x){
        //stable implementation of std::log(1.0 + std::exp(x)) for large values
        return std::log(1.0 + std::exp(-1.0 * std::abs(x))) + std::fmax(x,0.0);
    }

    static double Derivative(double x){
        return 1.0 / (1.0 + std::exp(-1.0 * x));
    }

    static double SecondDerivative(double x){
        return 1.0 / (2.0 + std::exp(-1.0 * x) + std::exp(x));
    }

    static double Inverse(double x){
        return std::fmin(std::log(std::exp(x) - 1.0), x);
    }

};

/**
 * @brief Defines the exponential function \f$g(x) = \exp(x)\f$.
 */
class Exp{
public:

    static double Evaluate(double x){
        return std::exp(x);
    }

    static double Derivative(double x){
        return std::exp(x);
    }

    static double SecondDerivative(double x){
        return std::exp(x);
    }

    static double Inverse(double x){
        return std::log(x);
    }

};

} // namespace mpart

#endif