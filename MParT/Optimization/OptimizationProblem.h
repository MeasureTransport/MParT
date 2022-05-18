#ifndef MPART_OPTIMIZATIONPROBLEM_H
#define MPART_OPTIMIZATIONPROBLEM_H

#include <Kokkos_Core.hpp>

namespace mpart{

    /**
     * @brief Abstract interface for optimization problems.

        The optimization problem we consider is
        \f$
            \min_{x} f(x)
        \f$
     * 
     */
    class OptimizationProblem {
    public:

        /**
         * @brief This function evaluates the objective f(x)
         * 
         */
        virtual double Objective(Kokkos::View<double*> const& x) = 0;

        /**
         * @brief Computes the gradient \f$\nabla f(x)\f$ and returns the value of \f$f(x)\f$.
         * 
         * @param x 
         * @param grad 
         * @return double 
         */
        virtual double Gradient(Kokkos::View<double*> const& x, Kokkos::View<double*> grad) =0;

        /**
         * @brief Computes \f$H*v\f$
         * 
         */
        //virtual void HessianAction() = 0;
    }; 


}


#endif 