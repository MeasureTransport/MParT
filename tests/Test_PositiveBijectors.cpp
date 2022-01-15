#include <catch2/catch_all.hpp>

#include "MParT/PositiveBijectors.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing soft plus function.", "[SofPlus]" ) {

    const double floatTol = 1e-15;

    std::vector<double> xs{-1.0, -0.5, 0.0, 0.1, 1.0};

    for(auto& x : xs){
        double eval = SoftPlus::Evaluate(x);
        CHECK( eval == Approx(std::log(1.0+std::exp(x))) );
        CHECK( SoftPlus::Derivative(x) == Approx(std::exp(x) / (std::exp(x) + 1.0)) );
        CHECK( SoftPlus::SecondDerivative(x) == Approx(std::exp(x) / std::pow(std::exp(x) + 1.0, 2.0)) );
        CHECK( SoftPlus::Inverse(eval) == Approx(x) );
    }
}