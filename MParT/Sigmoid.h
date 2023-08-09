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
class Sigmoid: public ParameterizedFunctionBase<MemorySpace>
{
    public:
    Sigmoid(StridedVector<const double, MemorySpace> centers, StridedVector<const double, MemorySpace> widths): ParameterizedFunctionBase<MemorySpace>(1, 1, widths.extent(0))
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
        auto policy = Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space>(0,pts.extent(1));
        Kokkos::parallel_for(policy, KOKKOS_CLASS_LAMBDA (unsigned int pointInd) {
            out(0,pointInd) = SigmoidType::Evaluate(pts(0,pointInd));
        });
    }
    void GradientImpl(StridedMatrix<const double, MemorySpace> const& sens, StridedMatrix<const double, MemorySpace> const& pts, StridedMatrix<double, MemorySpace> out) override {}
    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& sens,StridedMatrix<const double, MemorySpace> const& pts, StridedMatrix<double, MemorySpace> out) override {}

    private:

};
}

#endif //MPART_SIGMOID_H