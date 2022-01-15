#include <catch2/catch_all.hpp>

#include "MParT/OrthogonalPolynomial.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing Probabilist Hermite polynomials", "[ProbabilistHermite]" ) {

    const double floatTol = 1e-15;

    ProbabilistHermite poly;

    std::vector<double> xs{-1.0, -0.5, 0.0, 0.1, 1.0};
    std::vector<double> allvals(5);

    for(auto& x : xs){
        CHECK( poly.Evaluate(0, x) == 1.0 ); 
        CHECK( poly.Evaluate(1, x) == Approx(x).epsilon(floatTol) );
        CHECK( poly.Evaluate(2, x) == Approx(x*x-1.0).epsilon(floatTol) );
        CHECK( poly.Evaluate(3, x) == Approx(x*x*x-3.0*x).epsilon(floatTol) );
        CHECK( poly.Evaluate(4, x) == Approx(x*x*x*x - 6.0*x*x + 3.0).epsilon(floatTol) );

        poly.EvaluateAll(&allvals[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx(x*x-1.0).epsilon(floatTol) );
        CHECK( allvals[3] == Approx(x*x*x-3.0*x).epsilon(floatTol) );
        CHECK( allvals[4] == Approx(x*x*x*x - 6.0*x*x + 3.0).epsilon(floatTol) );
    }
}


TEST_CASE( "Testing Physicist Hermite polynomials", "[PhysicistHermite]" ) {

    const double floatTol = 1e-15;

    PhysicistHermite poly;

    std::vector<double> xs{-1.0, -0.5, 0.0, 0.1, 1.0};
    std::vector<double> allvals(5);


    for(auto& x : xs){
        CHECK( poly.Evaluate(0, x) == 1.0 ); 
        CHECK( poly.Evaluate(1, x) == Approx(2.0*x).epsilon(floatTol) );
        CHECK( poly.Evaluate(2, x) == Approx(4.0*x*x-2.0).epsilon(floatTol) );
        CHECK( poly.Evaluate(3, x) == Approx(8.0*x*x*x-12.0*x).epsilon(floatTol) );
        CHECK( poly.Evaluate(4, x) == Approx(16.0*x*x*x*x - 48.0*x*x + 12.0).epsilon(floatTol) );

        poly.EvaluateAll(&allvals[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(2.0*x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx(4.0*x*x-2.0).epsilon(floatTol) );
        CHECK( allvals[3] == Approx(8.0*x*x*x-12.0*x).epsilon(floatTol) );
        CHECK( allvals[4] == Approx(16.0*x*x*x*x - 48.0*x*x + 12.0).epsilon(floatTol) );
    }
}