#include <catch2/catch_all.hpp>

#include "MParT/Quadrature.h"

using namespace mpart;
using namespace Catch;


class TestIntegrand {
public:

    double operator()(double x) const{
        return std::exp(x);
    }

}; // class TestIntegrand



TEST_CASE( "Testing Recursive Quadrature", "[RecursiveQuadrature]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int maxSub = 10;
    double relTol = 1e-10;
    double absTol = 1e-10;

    // Set tolerance for tests
    double testTol = 1e-8;

    RecursiveQuadrature quad(maxSub,absTol,relTol);

    SECTION("Class Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        TestIntegrand integrand;
        double integral = quad.Integrate(integrand, lb, ub);

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

    SECTION("Lambda Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        auto integrand = [](double x){return exp(x);};
        double integral = quad.Integrate(integrand, lb, ub);    

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }
    
}


