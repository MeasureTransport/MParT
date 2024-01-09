#ifndef MPART_SIGMOID_H
#define MPART_SIGMOID_H

#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/Utilities/MathFunctions.h"

#include <Kokkos_Core.hpp>

namespace mpart{


/**
 * @brief A small namespace to store univariate functions used in @ref Sigmoid1d
 * 
 * A "sigmoid" class is expected to have at least three functions:
 * 
 * - \c Evaluate
 * - \c Derivative
 * - \c SecondDerivative
 */
namespace SigmoidTypes {
struct Logistic {
    KOKKOS_INLINE_FUNCTION double static Evaluate(double x) {
        return 0.5+0.5*MathSpace::tanh(x/2);
    }
    KOKKOS_INLINE_FUNCTION double static Inverse(double y) {
        return y > 1 ? -MathSpace::log((1-y)/y) : MathSpace::log(y/(1-y));
    }
    KOKKOS_INLINE_FUNCTION double static Derivative(double x) {
        double fx = Evaluate(x);
        return fx*(1-fx); // Known expression for the derivative of this
    }
    KOKKOS_INLINE_FUNCTION double static SecondDerivative(double x) {
        double fx = Evaluate(x);
        return fx*(1-fx)*(1-2*fx); // Known expression for the second derivative of this
    }
};
}

/**
 * @brief Class to represent univariate function space spanned by sigmoids
 * 
 * @details This class stores functions of the form 
 * \f[f_k(x;\mathbf{c},\mathbf{b},\mathbf{w})=b_0+b_1(x-c_1)+\sum_{j=2}^kw_js(b_j(x-c_j))\f]
 * 
 * where \f$w_j,b_j\geq 0\f$ and \f$s\f$ is a monotone increasing function.
 * While it is expected that \f$s\f$ is a sigmoid (e.g., Logistic or Erf
 * function), any monotone function will allow this class to work as expected.
 * Without loss of generality we set \f$b_0,b_1\equiv 1\f$ and
 * \f$c_1\equiv0\f$-- for a linear basis, this will be equivalent to the class
 * of functions presented above. If \f$k\in\{0,1\}\f$, then we revert only to
 * (increasing) linear functions.
 * 
 * In the above expression, we denote \f$\mathbf{c}\f$ as the "centers",
 * \f$\mathbf{b}\f$ as the "width" or "bandwidth", and \f$\mathbf{w}\f$ as the
 * "weights". If we want a function class that works with up to "order n"
 * sigmoids, then we need to store one "center+width+weight" for the first
 * order, then two "center+width+weight"s for the second order, then three
 * "center+width+weight"s for the third order and so on. Generally, the weights
 * will be uniform, but the centers and widths of these sigmoids can affect
 * behavior dramatically. If performing approximation in \f$L^2(\mu)\f$, a good
 * heuristic is to place the weights at the quantiles of \f$\mu\f$ and create
 * widths that are half the distance to the nearest neighbor.
 * 
 * @todo Currently, \f$b_0,b_1,c_1\f$ are not accounted for in the calculation.
 * 
 * @tparam MemorySpace Where the (nonlinear) parameters are stored
 * @tparam SigmoidType Class defining eval, @ref SigmoidTypes
 */
template<typename MemorySpace, typename SigmoidType>
class Sigmoid1d
{
    public:
    /**
     * @brief Construct a new Sigmoid 1d object
     * 
     * Each input should be of length \f$n(n+1)/2\f$, where \f$n\f$ is the
     * maximum order.
     * 
     * @param centers Where to center the sigmoids
     * @param widths How "wide" the sigmoids should be
     * @param weights How much to weight the sigmoids linearly
     */
    Sigmoid1d(Kokkos::View<double*, MemorySpace> centers,
              Kokkos::View<double*, MemorySpace> widths,
              Kokkos::View<double*, MemorySpace> weights):
        centers_(centers), widths_(widths), weights_(weights)
    {
        if(centers.extent(0) != widths.extent(0) || centers.extent(0) != weights.extent(0)) {
            std::stringstream ss;
            ss << "Sigmoid: incompatible dims of centers and widths.\n";
            ss << "centers: " << centers.extent(0) << ", \n";
            ss << "widths: " << widths.extent(0) << ",\n";
            ss << "weights: " << weights.extent(0) << "\n";
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
        // Arithmetic sum length calculation
        double order_double = (MathSpace::sqrt(1+8*centers.extent(0))-1)/2;
        int order = order_double;
        if(order <= 0 || MathSpace::abs((double)order - order_double) > 1e-15) {
            std::stringstream ss;
            ss << "Incorrect length of centers/widths/weights.";
            ss << "Length should be of form 1+2+3+...+n for some order n";
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
        order_ = order;
    }

    /**
     * @brief Construct a new Sigmoid 1d object from centers and widths
     * 
     * Each input should be of length \f$n(n+1)/2\f$, where \f$n\f$ is the
     * maximum order.
     * 
     * @param centers
     * @param widths 
     */
    Sigmoid1d(Kokkos::View<double*, MemorySpace> centers,
              Kokkos::View<double*, MemorySpace> widths):
        centers_(centers), widths_(widths)
    {
        Kokkos::View<double*,MemorySpace> weights ("Sigmoid weights", centers.extent(0));
        if(centers.extent(0) != widths.extent(0)) {
            std::stringstream ss;
            ss << "Sigmoid: incompatible dims of centers and widths.\n";
            ss << "centers: " << centers.extent(0) << ", \n";
            ss << "widths: " << widths.extent(0);
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
        // Arithmetic sum length calculation
        double order_double = (MathSpace::sqrt(1+8*centers.extent(0))-1)/2;
        int order = order_double;
        if(order <= 0 || MathSpace::abs((double)order - order_double) > 1e-15) {
            std::stringstream ss;
            ss << "Incorrect length of centers/widths/weights, " << centers.extent(0) << ".";
            ss << "Length should be of form 1+2+3+...+n for some order n>0";
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
        Kokkos::parallel_for(centers.extent(0), KOKKOS_LAMBDA(unsigned int i){weights(i) = 1.;});
        order_ = order;
    }

    /**
     * @brief Evaluate all sigmoids at one input
     * 
     * @param output Place to put the output (size max_order+1)
     * @param max_order Maximum order of basis function to evaluate
     * @param input Point to evaluate function
     */
    void EvaluateAll(double* output, int max_order, double input) const {
        if(order_ < max_order) {
            std::stringstream ss;
            ss << "Sigmoid basis evaluation order too large.\n";
            ss << "Given order " << max_order << ", ";
            ss << "can only evaluate up to order " << order_;
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
        
        int param_idx = 0;
        for(int curr_order = 0; curr_order <= max_order; curr_order++) {
            output[curr_order] = 0.;
            for(int basis_idx = 0; basis_idx < curr_order; basis_idx++) {
                output[curr_order] += weights_(param_idx)*SigmoidType::Evaluate(widths_(param_idx)*(input - centers_(param_idx)));
                param_idx++;
            }
        }
    }

    /**
     * @brief Evaluate all sigmoids up to given order and first derivatives
     * 
     * @param output Storage for sigmoid evaluation, size max_order+1
     * @param output_diff Storage for sigmoid derivative, size max_order+1
     * @param max_order Number of sigmoids to evaluate
     * @param input Where to evaluate sigmoids
     */
    void EvaluateDerivatives(double* output, double* output_diff, int max_order, double input) const {
        if(order_ < max_order) {
            std::stringstream ss;
            ss << "Sigmoid basis evaluation order too large.\n";
            ss << "Given order " << max_order << ", ";
            ss << "can only evaluate up to order " << order_;
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
        int param_idx = 0;
        for(int curr_order = 0; curr_order <= max_order; curr_order++) {
            output[curr_order] = 0.;
            output_diff[curr_order] = 0.;
            for(int basis_idx = 0; basis_idx < curr_order; basis_idx++) {
                output[curr_order] += weights_(param_idx)*SigmoidType::Evaluate(widths_(param_idx)*(input - centers_(param_idx)));
                output_diff[curr_order] += weights_(param_idx)*widths_(param_idx)*SigmoidType::Derivative(widths_(param_idx)*(input - centers_(param_idx)));
                param_idx++;
            }
        }
    }

    /**
     * @brief Evaluate sigmoids up to given order and first+second derivatives
     * 
     * @param output Storage for sigmoid evaluation, size max_order+1
     * @param output_diff Storage for sigmoid derivative, size max_order+1
     * @param output_diff2 Storage for sigmoid 2nd deriv, size max_order+1
     * @param max_order Maximum order of sigmoid to evaluate
     * @param input Where to evaluate the sigmoids
     */
    void EvaluateSecondDerivatives(double* output, double* output_diff, double* output_diff2, int max_order, double input) const {
        if(order_ < max_order) {
            std::stringstream ss;
            ss << "Sigmoid basis evaluation order too large.\n";
            ss << "Given order " << max_order << ", ";
            ss << "can only evaluate up to order " << order_;
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
        output[0] = 0.;
        output_diff[0] = 0.;
        output_diff2[0] = 0.;
        int param_idx = 0;
        for(int curr_order = 0; curr_order <= max_order; curr_order++) {
            output[curr_order] = 0.;
            output_diff[curr_order] = 0.;
            output_diff2[curr_order] = 0.;
            for(int basis_idx = 0; basis_idx < curr_order; basis_idx++) {
                output[curr_order] += weights_(param_idx)*SigmoidType::Evaluate(widths_(param_idx)*(input - centers_(param_idx)));
                output_diff[curr_order] += weights_(param_idx)*widths_(param_idx)*SigmoidType::Derivative(widths_(param_idx)*(input - centers_(param_idx)));
                output_diff2[curr_order] += weights_(param_idx)*widths_(param_idx)*widths_(param_idx)*SigmoidType::SecondDerivative(widths_(param_idx)*(input - centers_(param_idx)));
                param_idx++;
            }
        }
    }

    /**
     * @brief Get the maximum order of this function class
     * 
     * @return int 
     */
    int GetOrder() const {return order_;}

    private:
    int order_;
    Kokkos::View<const double*, MemorySpace> centers_, widths_, weights_;
};
}

#endif //MPART_SIGMOID_H