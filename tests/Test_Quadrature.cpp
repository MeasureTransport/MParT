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

TEST_CASE( "Testing CC Quadrature", "[ClenshawCurtisQuadrature]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int order = 10;

    // Set tolerance for tests
    double testTol = 1e-8;

    ClenshawCurtisQuadrature quad(order);

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


TEST_CASE( "Testing Recursive Quadrature", "[RecursiveQuadrature]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int maxSub = 10;
    double relTol = 1e-7;
    double absTol = 1e-7;
    unsigned int order = 8;

    // Set tolerance for tests
    double testTol = 1e-4;

    RecursiveQuadrature quad(maxSub, order, absTol, relTol);

    SECTION("Class Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        TestIntegrand integrand;
        double integral = quad.Integrate(integrand, lb, ub);

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( quad.Status()>0 );
        CHECK( quad.MaxLevel()<maxSub );
    }

    SECTION("Lambda Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        auto integrand = [](double x){return exp(x);};
        double integral = quad.Integrate(integrand, lb, ub);    

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( quad.Status()>0 );
        CHECK( quad.MaxLevel()<maxSub );
    }


    SECTION("Discontinuous Integrand")
    {   
        double lb = 0;
        double ub = 1.0;
        unsigned int numEvals = 0;

        auto integrand = [&](double x){
            numEvals++;
            if(x<0.5)
                return exp(x);
            else 
                return 1.0+exp(x);
        };
        double integral = quad.Integrate(integrand, lb, ub);    

        double trueVal = (ub-0.5) + exp(ub)-exp(lb);
        CHECK( integral == Approx(trueVal).epsilon(testTol) );
        CHECK( quad.Status()>0 );
        CHECK( quad.MaxLevel()<=maxSub );
        CHECK( numEvals<400);
    }
    
}



TEST_CASE( "Testing Adaptive Simpson Integration", "[AdaptiveSimpson]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int maxSub = 30;
    double relTol = 1e-7;
    double absTol = 1e-7;

    // Set tolerance for tests
    double testTol = 1e-4;

    AdaptiveSimpson quad(maxSub, absTol, relTol);

    SECTION("Class Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        TestIntegrand integrand;
        double integral = quad.Integrate(integrand, lb, ub);

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( quad.Status()>0 );
        CHECK( quad.MaxLevel()<maxSub );
    }

    SECTION("Lambda Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        auto integrand = [](double x){return exp(x);};
        double integral = quad.Integrate(integrand, lb, ub);    

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( quad.Status()>0 );
        CHECK( quad.MaxLevel()<maxSub );
    }


    SECTION("Discontinuous Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        unsigned int numEvals = 0;

        auto integrand = [&](double x){
            numEvals++;
            if(x<0.5)
                return exp(x);
            else 
                return 1.0+exp(x);
        };
        double integral = quad.Integrate(integrand, lb, ub);    

        double trueVal = (ub-0.5) + exp(ub)-exp(lb);
        CHECK( integral == Approx(trueVal).epsilon(testTol) );
        CHECK( quad.Status()>0 );
        CHECK( quad.MaxLevel()<=maxSub );
        CHECK( numEvals<150);
    }
    
}

