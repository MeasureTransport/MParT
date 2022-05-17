#ifndef MPART_GRADIENTDESCENT_H
#define MPART_GRADIENTDESCENT_H

#include <Kokkos_Core.hpp>


namespace mpart{

class GradientDescent : public LineSearchOptimizer {
public:

    virtual void StepDirection(Kokkos::View<double*> x, Kokkos::View<double*> dir) override
    {
        prob_->Gradient(x,dir);
    }

};

}

#endif 