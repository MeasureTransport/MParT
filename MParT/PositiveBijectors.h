#ifndef MPART_POSITIVEBIJECTORS_H
#define MPART_POSITIVEBIJECTORS_H

#include <Kokkos_Core.hpp>
#include <math.h>

namespace mpart{

/**
 * @brief Defines the softplus function \f$g(x) = \log(1+\exp(x))\f$.
 */
class SoftPlus{
public:

    KOKKOS_INLINE_FUNCTION static double Evaluate(double x){
        //stable implementation of std::log(1.0 + std::exp(x)) for large values
        return std::log(1.0 + std::exp(-1.0 * std::abs(x))) + std::fmax(x,0.0);
    }

    KOKKOS_INLINE_FUNCTION static double Derivative(double x){
        return x < 0 ? std::exp(x) / (std::exp(x) + 1.0) : 1.0 / (1.0 + std::exp(-1.0 * x));
    }

    KOKKOS_INLINE_FUNCTION static double SecondDerivative(double x){
        double fx = Derivative(x);
        return fx * (1.0 - fx);
    }

    KOKKOS_INLINE_FUNCTION static double Inverse(double x){
        return std::fmin(std::log(std::exp(x) - 1.0), x);
    }

};

/**
 * @brief Defines the exponential function \f$g(x) = \exp(x)\f$.
 */
class Exp{
public:

    KOKKOS_INLINE_FUNCTION static double Evaluate(double x){
        return std::exp(x);
    }

    KOKKOS_INLINE_FUNCTION static double Derivative(double x){
        return std::exp(x);
    }

    KOKKOS_INLINE_FUNCTION static double SecondDerivative(double x){
        return std::exp(x);
    }

    KOKKOS_INLINE_FUNCTION static double Inverse(double x){
        return std::log(x);
    }

};

} // namespace mpart

#endif