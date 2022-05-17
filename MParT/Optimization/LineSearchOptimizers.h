#ifndef MPART_OPTIMIZATIONPROBLEM_H
#define MPART_OPTIMIZATIONPROBLEM_H

#include <Kokkos_Core.hpp>


namespace mpart{

class OptimizationResult {
public:

    Kokkos::View<double*> xopt;
    double fopt;

    Kokkos::View<double*> objHistory;
};

class LineSearchOptimizer {
public:

    LineSearchOptimizer(std::shared_ptr<OptimizationProblem> const& prob) : prob_(prob){};

    /**
     * @brief Computes a search step at the current point x.  
     * 
     * @param x 
     * @param dir 
     */
    virtual void StepDirection(Kokkos::View<double*> x, Kokkos::View<double*> dir) = 0;

    /**
     * @brief Computes a backtracking line search to make sure the Armijo-Wolfe conditions are satisfied.
     * 
     * @param xcurr Current point 
     * @param dir Search direction at current point
     * @param xnew New point, computed by this function, that satisties the AW conditions.
     */
    void LineSearch(Kokkos::View<double*> xcurr, Kokkos::View<double*> dir, Kokkos::View<double*> xnew);

    /**
     * @brief Solves the optimization problem.
     * 
     * @param x0 Initial guess.
     */
    OptimizationResult Solve(Kokkos::View<double*> x0);

protected:
    
    std::shared_ptr<OptimizationProblem> const& prob_;

};


}

#endif 