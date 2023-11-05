#ifndef MPART_SIGMOID_H
#define MPART_SIGMOID_H

#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/Miscellaneous.h"

#include <Kokkos_Core.hpp>

namespace mpart{

#if (KOKKOS_VERSION / 10000 == 3) && (KOKKOS_VERSION / 100 % 100 < 7)
namespace MathSpace = Kokkos::Experimental;
#else
namespace MathSpace = Kokkos;
#endif

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
};
}

template<typename MemorySpace, typename SigmoidType>
class Sigmoid1d
{
    public:
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


    void EvaluateAll(double* output, int max_order, double input) {
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

    void EvaluateDerivatives(double* output, double* output_diff, int max_order, double input) {
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

    void EvaluateSecondDerivatives(double* output, double* output_diff, double* output_diff2, int max_order, double input) {
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
                output_diff2[curr_order] += weights_(param_idx)*widths_(param_idx)*SigmoidType::Derivative(widths_(param_idx)*(input - centers_(param_idx)));
                param_idx++;
            }
        }
    }

    int GetOrder() const {return order_;}

    private:
    int order_;
    Kokkos::View<const double*, MemorySpace> centers_, widths_, weights_;
};
}

#endif //MPART_SIGMOID_H