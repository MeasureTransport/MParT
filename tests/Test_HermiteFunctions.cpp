#include <catch2/catch_all.hpp>

#include "MParT/HermiteFunction.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing Hermite functions", "[HermiteFunction]" ) {

    const double floatTol = 1e-15;

    HermiteFunction poly;

    std::vector<double> xs{-1.0, -0.5, 0.0, 0.1, 1.0};
    std::vector<double> allvals(5);

    double truth;

    for(auto& x : xs){
        poly.EvaluateAll(&allvals[0], 4, x);
        
        truth = std::pow(M_PI, -0.25) * std::exp(-0.5*x*x);
        CHECK( poly.Evaluate(0, x) == Approx(truth).epsilon(floatTol) ); 
        CHECK( allvals[0] == Approx(truth).epsilon(floatTol));

        truth = std::sqrt(2.0)*std::pow(M_PI, -0.25) * x*std::exp(-0.5*x*x);
        CHECK( poly.Evaluate(1, x) == Approx(truth).epsilon(floatTol) ); 
        CHECK( allvals[1] == Approx(truth).epsilon(floatTol));
        

        truth = std::pow(2.0, -0.5) * std::pow(M_PI, -0.25) * (2.0*x*x-1.0) * std::exp(-0.5*x*x);
        CHECK( poly.Evaluate(2, x) == Approx(truth).epsilon(floatTol) ); 
        CHECK( allvals[2] == Approx(truth).epsilon(floatTol));
        

        truth = std::pow(3.0, -0.5) * std::pow(M_PI, -0.25) * (2.0*x*x*x-3.0*x) * std::exp(-0.5*x*x);
        CHECK( poly.Evaluate(3, x) == Approx(truth).epsilon(floatTol) ); 
        CHECK( allvals[3] == Approx(truth).epsilon(floatTol));
        

        truth = std::pow(24.0, -0.5) * std::pow(M_PI, -0.25) * (4.0*x*x*x*x - 12*x*x + 3.0) * std::exp(-0.5*x*x);
        CHECK( poly.Evaluate(4, x) == Approx(truth).epsilon(floatTol) ); 
        CHECK( allvals[4] == Approx(truth).epsilon(floatTol));
    }
}