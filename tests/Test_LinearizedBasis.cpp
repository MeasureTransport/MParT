#include <catch2/catch_all.hpp>

#include "MParT/OrthogonalPolynomial.h"
#include "MParT/LinearizedBasis.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing Linearized hermite basis", "[LinearizedHermite]" ) {

    const double floatTol = 1e-15;

    double lb = -3;
    double ub = 3;

    LinearizedBasis<ProbabilistHermite> basis(-3,3);

    std::vector<double> xs1{-1.9, -0.25, 0.0, 0.5, 1.9};
    std::vector<double> xs2{-5.0,-4.0,-3.1};
    std::vector<double> xs3{3.1,4.0,5.0};
    std::vector<double> allvals(5);
    std::vector<double> allderivs(5);
    std::vector<double> allderivs2(5);
    
    // Check the evaluation
    for(auto& x : xs1){
        CHECK( basis.Evaluate(0, x) == 1.0 ); 
        CHECK( basis.Evaluate(1, x) == Approx(x).epsilon(floatTol) );
        CHECK( basis.Evaluate(2, x) == Approx(x*x-1.0).epsilon(floatTol) );
        CHECK( basis.Evaluate(3, x) == Approx(x*x*x-3.0*x).epsilon(floatTol) );
        CHECK( basis.Evaluate(4, x) == Approx(x*x*x*x - 6.0*x*x + 3.0).epsilon(floatTol) );

        basis.EvaluateAll(&allvals[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx(x*x-1.0).epsilon(floatTol) );
        CHECK( allvals[3] == Approx(x*x*x-3.0*x).epsilon(floatTol) );
        CHECK( allvals[4] == Approx(x*x*x*x - 6.0*x*x + 3.0).epsilon(floatTol) );
    }


    // Check the evaluation outside the bounds
    for(auto& x : xs2){
        CHECK( basis.Evaluate(0, x) == 1.0 ); 
        CHECK( basis.Evaluate(1, x) == Approx(x).epsilon(floatTol) );
        CHECK( basis.Evaluate(2, x) == Approx((lb*lb-1.0) + 2.0*lb*(x-lb)).epsilon(floatTol) );
        CHECK( basis.Evaluate(3, x) == Approx((lb*lb*lb-3.0*lb) + (3.0*lb*lb-3.0)*(x-lb)).epsilon(floatTol) );
        CHECK( basis.Evaluate(4, x) == Approx((lb*lb*lb*lb - 6.0*lb*lb + 3.0) + (4.0*lb*lb*lb - 12*lb)*(x-lb)).epsilon(floatTol) );

        basis.EvaluateAll(&allvals[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx((lb*lb-1.0) + 2.0*lb*(x-lb)).epsilon(floatTol) );
        CHECK( allvals[3] == Approx((lb*lb*lb-3.0*lb) + (3.0*lb*lb-3.0)*(x-lb)).epsilon(floatTol) );
        CHECK( allvals[4] == Approx((lb*lb*lb*lb - 6.0*lb*lb + 3.0) + (4.0*lb*lb*lb - 12*lb)*(x-lb)).epsilon(floatTol) );
    }

    // Check the evaluation outside the bounds
    for(auto& x : xs3){
        CHECK( basis.Evaluate(0, x) == 1.0 ); 
        CHECK( basis.Evaluate(1, x) == Approx(x).epsilon(floatTol) );
        CHECK( basis.Evaluate(2, x) == Approx((ub*ub-1.0) + 2.0*ub*(x-ub)).epsilon(floatTol) );
        CHECK( basis.Evaluate(3, x) == Approx((ub*ub*ub-3.0*ub) + (3.0*ub*ub-3.0)*(x-ub)).epsilon(floatTol) );
        CHECK( basis.Evaluate(4, x) == Approx((ub*ub*ub*ub - 6.0*ub*ub + 3.0) + (4.0*ub*ub*ub - 12*ub)*(x-ub)).epsilon(floatTol) );

        basis.EvaluateAll(&allvals[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx((ub*ub-1.0) + 2.0*ub*(x-ub)).epsilon(floatTol) );
        CHECK( allvals[3] == Approx((ub*ub*ub-3.0*ub) + (3.0*ub*ub-3.0)*(x-ub)).epsilon(floatTol) );
        CHECK( allvals[4] == Approx((ub*ub*ub*ub - 6.0*ub*ub + 3.0) + (4.0*ub*ub*ub - 12*ub)*(x-ub)).epsilon(floatTol) );
    }

    // Check the derivative
    for(auto& x : xs1){
        CHECK( basis.Derivative(0, x) == 0.0 ); 
        CHECK( basis.Derivative(1, x) == 1.0 );
        CHECK( basis.Derivative(2, x) == Approx(2.0*x).epsilon(floatTol) );
        CHECK( basis.Derivative(3, x) == Approx(3.0*x*x-3.0).epsilon(floatTol) );
        CHECK( basis.Derivative(4, x) == Approx(4.0*x*x*x - 12.0*x).epsilon(floatTol) );

        basis.EvaluateDerivatives(&allvals[0], &allderivs[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx(x*x-1.0).epsilon(floatTol) );
        CHECK( allvals[3] == Approx(x*x*x-3.0*x).epsilon(floatTol) );
        CHECK( allvals[4] == Approx(x*x*x*x - 6.0*x*x + 3.0).epsilon(floatTol) );
        CHECK( allderivs[0] == 0.0 ); 
        CHECK( allderivs[1] == 1.0 );
        CHECK( allderivs[2] == Approx(2.0*x).epsilon(floatTol) );
        CHECK( allderivs[3] == Approx(3.0*x*x-3.0).epsilon(floatTol) );
        CHECK( allderivs[4] == Approx(4.0*x*x*x - 12.0*x).epsilon(floatTol) );
    }

    // Check the derivative
    for(auto& x : xs2){
        CHECK( basis.Derivative(0, x) == 0.0 ); 
        CHECK( basis.Derivative(1, x) == 1.0 );
        CHECK( basis.Derivative(2, x) == Approx(2.0*lb).epsilon(floatTol) );
        CHECK( basis.Derivative(3, x) == Approx(3.0*lb*lb-3.0).epsilon(floatTol) );
        CHECK( basis.Derivative(4, x) == Approx(4.0*lb*lb*lb - 12.0*lb).epsilon(floatTol) );

        basis.EvaluateDerivatives(&allvals[0], &allderivs[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx((lb*lb-1.0) + 2.0*lb*(x-lb)).epsilon(floatTol) );
        CHECK( allvals[3] == Approx((lb*lb*lb-3.0*lb) + (3.0*lb*lb-3.0)*(x-lb)).epsilon(floatTol) );
        CHECK( allvals[4] == Approx((lb*lb*lb*lb - 6.0*lb*lb + 3.0) + (4.0*lb*lb*lb - 12*lb)*(x-lb)).epsilon(floatTol) );
        CHECK( allderivs[0] == 0.0 ); 
        CHECK( allderivs[1] == 1.0 );
        CHECK( allderivs[2] == Approx(2.0*lb).epsilon(floatTol) );
        CHECK( allderivs[3] == Approx(3.0*lb*lb-3.0).epsilon(floatTol) );
        CHECK( allderivs[4] == Approx(4.0*lb*lb*lb - 12.0*lb).epsilon(floatTol) );
    }

    for(auto& x : xs3){
        CHECK( basis.Derivative(0, x) == 0.0 ); 
        CHECK( basis.Derivative(1, x) == 1.0 );
        CHECK( basis.Derivative(2, x) == Approx(2.0*ub).epsilon(floatTol) );
        CHECK( basis.Derivative(3, x) == Approx(3.0*ub*ub-3.0).epsilon(floatTol) );
        CHECK( basis.Derivative(4, x) == Approx(4.0*ub*ub*ub - 12.0*ub).epsilon(floatTol) );

        basis.EvaluateDerivatives(&allvals[0], &allderivs[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx((ub*ub-1.0) + 2.0*ub*(x-ub)).epsilon(floatTol) );
        CHECK( allvals[3] == Approx((ub*ub*ub-3.0*ub) + (3.0*ub*ub-3.0)*(x-ub)).epsilon(floatTol) );
        CHECK( allvals[4] == Approx((ub*ub*ub*ub - 6.0*ub*ub + 3.0) + (4.0*ub*ub*ub - 12*ub)*(x-ub)).epsilon(floatTol) );
        CHECK( allderivs[0] == 0.0 ); 
        CHECK( allderivs[1] == 1.0 );
        CHECK( allderivs[2] == Approx(2.0*ub).epsilon(floatTol) );
        CHECK( allderivs[3] == Approx(3.0*ub*ub-3.0).epsilon(floatTol) );
        CHECK( allderivs[4] == Approx(4.0*ub*ub*ub - 12.0*ub).epsilon(floatTol) );
    }

    // Check the second derivatives
    for(auto& x : xs1){
        CHECK( basis.SecondDerivative(0, x) == 0.0 ); 
        CHECK( basis.SecondDerivative(1, x) == 0.0 );
        CHECK( basis.SecondDerivative(2, x) == Approx(2.0).epsilon(floatTol) );
        CHECK( basis.SecondDerivative(3, x) == Approx(6.0*x).epsilon(floatTol) );
        CHECK( basis.SecondDerivative(4, x) == Approx(12.0*x*x - 12.0).epsilon(floatTol) );

        basis.EvaluateSecondDerivatives(&allvals[0], &allderivs[0], &allderivs2[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx(x*x-1.0).epsilon(floatTol) );
        CHECK( allvals[3] == Approx(x*x*x-3.0*x).epsilon(floatTol) );
        CHECK( allvals[4] == Approx(x*x*x*x - 6.0*x*x + 3.0).epsilon(floatTol) );
        CHECK( allderivs[0] == 0.0 ); 
        CHECK( allderivs[1] == 1.0 );
        CHECK( allderivs[2] == Approx(2.0*x).epsilon(floatTol) );
        CHECK( allderivs[3] == Approx(3.0*x*x-3.0).epsilon(floatTol) );
        CHECK( allderivs[4] == Approx(4.0*x*x*x - 12.0*x).epsilon(floatTol) );
        CHECK( allderivs2[0] == 0.0 ); 
        CHECK( allderivs2[1] == 0.0 );
        CHECK( allderivs2[2] == Approx(2.0).epsilon(floatTol) );
        CHECK( allderivs2[3] == Approx(6.0*x).epsilon(floatTol) );
        CHECK( allderivs2[4] == Approx(12.0*x*x - 12.0).epsilon(floatTol) );
    }

    for(auto& x : xs2){
        CHECK( basis.SecondDerivative(0, x) == 0.0 ); 
        CHECK( basis.SecondDerivative(1, x) == 0.0 );
        CHECK( basis.SecondDerivative(2, x) == 0.0 );
        CHECK( basis.SecondDerivative(3, x) == 0.0 );
        CHECK( basis.SecondDerivative(4, x) == 0.0 );

        basis.EvaluateSecondDerivatives(&allvals[0], &allderivs[0], &allderivs2[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx((lb*lb-1.0) + 2.0*lb*(x-lb)).epsilon(floatTol) );
        CHECK( allvals[3] == Approx((lb*lb*lb-3.0*lb) + (3.0*lb*lb-3.0)*(x-lb)).epsilon(floatTol) );
        CHECK( allvals[4] == Approx((lb*lb*lb*lb - 6.0*lb*lb + 3.0) + (4.0*lb*lb*lb - 12*lb)*(x-lb)).epsilon(floatTol) );
        CHECK( allderivs[0] == 0.0 ); 
        CHECK( allderivs[1] == 1.0 );
        CHECK( allderivs[2] == Approx(2.0*lb).epsilon(floatTol) );
        CHECK( allderivs[3] == Approx(3.0*lb*lb-3.0).epsilon(floatTol) );
        CHECK( allderivs[4] == Approx(4.0*lb*lb*lb - 12.0*lb).epsilon(floatTol) );
        CHECK( allderivs2[0] == 0.0 ); 
        CHECK( allderivs2[1] == 0.0 );
        CHECK( allderivs2[2] == 0.0 );
        CHECK( allderivs2[3] == 0.0 );
        CHECK( allderivs2[4] == 0.0 );
    }

    for(auto& x : xs3){
        CHECK( basis.SecondDerivative(0, x) == 0.0 ); 
        CHECK( basis.SecondDerivative(1, x) == 0.0 );
        CHECK( basis.SecondDerivative(2, x) == 0.0 );
        CHECK( basis.SecondDerivative(3, x) == 0.0 );
        CHECK( basis.SecondDerivative(4, x) == 0.0 );

        basis.EvaluateSecondDerivatives(&allvals[0], &allderivs[0], &allderivs2[0], 4, x);
        CHECK( allvals[0] == 1.0 );
        CHECK( allvals[1] == Approx(x).epsilon(floatTol) );
        CHECK( allvals[2] == Approx((ub*ub-1.0) + 2.0*ub*(x-ub)).epsilon(floatTol) );
        CHECK( allvals[3] == Approx((ub*ub*ub-3.0*ub) + (3.0*ub*ub-3.0)*(x-ub)).epsilon(floatTol) );
        CHECK( allvals[4] == Approx((ub*ub*ub*ub - 6.0*ub*ub + 3.0) + (4.0*ub*ub*ub - 12*ub)*(x-ub)).epsilon(floatTol) );
        CHECK( allderivs[0] == 0.0 ); 
        CHECK( allderivs[1] == 1.0 );
        CHECK( allderivs[2] == Approx(2.0*ub).epsilon(floatTol) );
        CHECK( allderivs[3] == Approx(3.0*ub*ub-3.0).epsilon(floatTol) );
        CHECK( allderivs[4] == Approx(4.0*ub*ub*ub - 12.0*ub).epsilon(floatTol) );
        CHECK( allderivs2[0] == 0.0 ); 
        CHECK( allderivs2[1] == 0.0 );
        CHECK( allderivs2[2] == 0.0 );
        CHECK( allderivs2[3] == 0.0 );
        CHECK( allderivs2[4] == 0.0 );
    }
}