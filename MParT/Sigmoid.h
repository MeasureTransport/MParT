#ifndef MPART_SIGMOID_H
#define MPART_SIGMOID_H

#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/ConditionalMapBase.h"

#include <Kokkos_Core.hpp>

namespace mpart{

#if (KOKKOS_VERSION / 10000 == 3) && (KOKKOS_VERSION / 100 % 100 < 7)
namespace MathSpace = Kokkos::Experimental;
#else
namespace MathSpace = Kokkos;
#endif

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

template<typename MemorySpace, typename SigmoidType>
class Sigmoid1d: public ParameterizedFunctionBase<MemorySpace>
{
    public:
    Sigmoid1d(StridedVector<const double, MemorySpace> centers, StridedVector<const double, MemorySpace> widths):
        centers_(centers), widths_(widths), ParameterizedFunctionBase<MemorySpace>(1, 1, widths.extent(0))
    {
        if(centers.extent(0) != widths.extent(0)) {
            std::stringstream ss;
            ss << "Sigmoid: incompatible dims of centers and widths.";
            ss << "centers: " << centers.extent(0) << ", ";
            ss << "widths: " << widths.extent(0);
            ProcAgnosticError<MemorySpace,std::invalid_argument>::error(ss.str().c_str());
        }
    }

    void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts, StridedMatrix<double, MemorySpace> out) override {
        Kokkos::parallel_for(pts.extent(1), KOKKOS_CLASS_LAMBDA (unsigned int sample_index) {
            double eval_pt = 0.;
            for(int coeff_index = 0; coeff_index < this->numCoeffs; coeff_index++){
                eval_pt += this->savedCoeffs(coeff_index)*SigmoidType::Evaluate(widths_(coeff_index)*(pts(0,sample_index)-centers_(coeff_index)));
            }
            out(0,sample_index) = eval_pt;
        });
    }

    void GradientImpl(StridedMatrix<const double, MemorySpace> const& sens, StridedMatrix<const double, MemorySpace> const& pts, StridedMatrix<double, MemorySpace> out) override {
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace> policy ({0,0},{this->numCoeffs, (int) pts.extent(1)});
        Kokkos::parallel_for(pts.extent(1), KOKKOS_CLASS_LAMBDA(unsigned int sample_index) {
            double grad_pt = 0.;
            for(int coeff_index = 0; coeff_index < this->numCoeffs; coeff_index++){
                grad_pt += this->savedCoeffs(coeff_index)*widths_(coeff_index)*SigmoidType::Derivative(widths_(coeff_index)*(pts(0,sample_index)-centers_(coeff_index)));
            }
            out(0,sample_index) = sens(0,sample_index)*grad_pt;
        });
    }

    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& sens,StridedMatrix<const double, MemorySpace> const& pts, StridedMatrix<double, MemorySpace> out) override {
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, ExecutionSpace> policy ({0,0}, {this->numCoeffs, (int)sens.extent(1)});
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA(unsigned int coeff_index, unsigned int sample_index) {
            out(coeff_index,sample_index) = sens(0,sample_index)*SigmoidType::Evaluate(widths_(coeff_index)*(pts(0,sample_index)-centers_(coeff_index)));
        });
    }

    private:
    using ExecutionSpace = typename MemoryToExecution<MemorySpace>::Space;
    Kokkos::View<const double*, MemorySpace> centers_, widths_;
};
}

#endif //MPART_SIGMOID_H