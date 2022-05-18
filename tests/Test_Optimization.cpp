#include <catch2/catch_all.hpp>

#include "MParT/Optimization/OptimizationProblem.h"


using namespace mpart;
using namespace Catch;

class RosenbrockTestObjective : public OptimizationProblem {
public:

    /**
        * @brief This function evaluates the objective f(x)
        * 
        */
    virtual double Objective(Kokkos::View<double*> const& x) override
    {
        return x(0) - x(1)*x(1);
    }

    /**
        * @brief Computes the gradient \f$\nabla f(x)\f$ and returns the value of \f$f(x)\f$.
        * 
        * @param x 
        * @param grad 
        * @return double 
        */
    virtual double Gradient(Kokkos::View<double*> const& x, Kokkos::View<double*> grad) override 
    {
        double obj = x(0) - x(1)*x(1);

        grad(0) = 1.0;
        grad(1) = -2.0*x(1);
        return obj;
    }

};

TEST_CASE( "Optimization problem basic test", "[BasicOptTest]") {

    RosenbrockTestObjective obj;

    Kokkos::View<double*> x("Test Point", 2);
    x(0) = 0.5;
    x(1) = 0.5;

    double testVal = obj.Objective(x);
    double trueVal = x(0) - x(1)*x(1);

    CHECK(testVal == Approx(trueVal).epsilon(1e-12));
}