#include <catch2/catch_all.hpp>

#include "MParT/UnivariateBases.h"
using namespace mpart;

using namespace Catch::Matchers;

TEST_CASE( "Testing Sine basis", "[SineBasis]" ) {

    const double floatTol = 1e-12;

    SineBasis sine;

    std::vector<double> xs{0.0, 0.1, 0.5, 0.75, 1.0};
    std::vector<double> allvals(5);
    std::vector<double> allderivs(5);
    std::vector<double> allderivs2(5);
    auto sine_evaluation = [](double x){return std::vector<double> {
            1.0,
            sin(2*M_PI*x),
            sin(2*M_PI*2*x),
            sin(2*M_PI*3*x),
            sin(2*M_PI*4*x)
        };
    };
    // Check the evaluation
    for(auto& x : xs){
        auto sine_eval = sine_evaluation(x);
        // CHECK_THAT(inv(0, i), WithinAbs(points(0, i), 1e-10));
        CHECK     ( sine.Evaluate(0, x) ==        sine_eval[0]             ); 
        CHECK_THAT( sine.Evaluate(1, x), WithinAbs(sine_eval[1], floatTol) );
        CHECK_THAT( sine.Evaluate(2, x), WithinAbs(sine_eval[2], floatTol) );
        CHECK_THAT( sine.Evaluate(3, x), WithinAbs(sine_eval[3], floatTol) );
        CHECK_THAT( sine.Evaluate(4, x), WithinAbs(sine_eval[4], floatTol) );

        sine.EvaluateAll(&allvals[0], 4, x);
        CHECK     ( allvals[0] ==        sine_eval[0]             );
        CHECK_THAT( allvals[1], WithinAbs(sine_eval[1], floatTol) );
        CHECK_THAT( allvals[2], WithinAbs(sine_eval[2], floatTol) );
        CHECK_THAT( allvals[3], WithinAbs(sine_eval[3], floatTol) );
        CHECK_THAT( allvals[4], WithinAbs(sine_eval[4], floatTol) );
    }
    
    auto sine_derivative = [](double x){return std::vector<double> {
            0.0,
            1*2*M_PI*cos(1*2*M_PI*x),
            2*2*M_PI*cos(2*2*M_PI*x),
            3*2*M_PI*cos(3*2*M_PI*x),
            4*2*M_PI*cos(4*2*M_PI*x)
        };
    };
    // CHECK_THAT the derivative
    for(auto& x : xs){
        auto sine_deriv = sine_derivative(x);
        CHECK     ( sine.Derivative(0, x) ==        sine_deriv[0] ); 
        CHECK     ( sine.Derivative(1, x) ==        sine_deriv[1] );
        CHECK_THAT( sine.Derivative(2, x), WithinAbs(sine_deriv[2], floatTol) );
        CHECK_THAT( sine.Derivative(3, x), WithinAbs(sine_deriv[3], floatTol) );
        CHECK_THAT( sine.Derivative(4, x), WithinAbs(sine_deriv[4], floatTol) );

        auto sine_eval = sine_evaluation(x);
        sine.EvaluateDerivatives(&allvals[0], &allderivs[0], 4, x);
        CHECK     ( allvals[0] ==        sine_eval[0] );
        CHECK_THAT( allvals[1], WithinAbs(sine_eval[1], floatTol) );
        CHECK_THAT( allvals[2], WithinAbs(sine_eval[2], floatTol) );
        CHECK_THAT( allvals[3], WithinAbs(sine_eval[3], floatTol) );
        CHECK_THAT( allvals[4], WithinAbs(sine_eval[4], floatTol) );
        CHECK     ( allderivs[0] ==        sine_deriv[0] ); 
        CHECK     ( allderivs[1] ==        sine_deriv[1] );
        CHECK_THAT( allderivs[2], WithinAbs(sine_deriv[2], floatTol) );
        CHECK_THAT( allderivs[3], WithinAbs(sine_deriv[3], floatTol) );
        CHECK_THAT( allderivs[4], WithinAbs(sine_deriv[4], floatTol) );

        sine.EvaluateDerivatives(&allderivs[0], 4, x);
        CHECK     ( allderivs[0] ==        sine_deriv[0] );
        CHECK     ( allderivs[1] ==        sine_deriv[1] );
        CHECK_THAT( allderivs[2], WithinAbs(sine_deriv[2], floatTol) );
        CHECK_THAT( allderivs[3], WithinAbs(sine_deriv[3], floatTol) );
        CHECK_THAT( allderivs[4], WithinAbs(sine_deriv[4], floatTol) );
    }
    
    auto sine_derivative2 = [](double x){return std::vector<double> {
            0.0,
            -(1*2*M_PI)*(1*2*M_PI)*sin(1*2*M_PI*x),
            -(2*2*M_PI)*(2*2*M_PI)*sin(2*2*M_PI*x),
            -(3*2*M_PI)*(3*2*M_PI)*sin(3*2*M_PI*x),
            -(4*2*M_PI)*(4*2*M_PI)*sin(4*2*M_PI*x)
        };
    };
    // CHECK_THAT the second derivatives
    for(auto& x : xs){
        auto sine_deriv2 = sine_derivative2(x);
        CHECK     ( sine.SecondDerivative(0, x) ==        sine_deriv2[0]             );
        CHECK     ( sine.SecondDerivative(1, x) ==        sine_deriv2[1]             );
        CHECK_THAT( sine.SecondDerivative(2, x), WithinAbs(sine_deriv2[2], floatTol) );
        CHECK_THAT( sine.SecondDerivative(3, x), WithinAbs(sine_deriv2[3], floatTol) );
        CHECK_THAT( sine.SecondDerivative(4, x), WithinAbs(sine_deriv2[4], floatTol) );

        auto sine_eval = sine_evaluation(x);
        auto sine_deriv = sine_derivative(x);
        sine.EvaluateSecondDerivatives(&allvals[0], &allderivs[0], &allderivs2[0], 4, x);
        CHECK     ( allvals[0] ==        sine_eval[0]             );
        CHECK_THAT( allvals[1], WithinAbs(sine_eval[1], floatTol) );
        CHECK_THAT( allvals[2], WithinAbs(sine_eval[2], floatTol) );
        CHECK_THAT( allvals[3], WithinAbs(sine_eval[3], floatTol) );
        CHECK_THAT( allvals[4], WithinAbs(sine_eval[4], floatTol) );
        CHECK     ( allderivs[0] ==        sine_deriv[0]             );
        CHECK     ( allderivs[1] ==        sine_deriv[1]             );
        CHECK_THAT( allderivs[2], WithinAbs(sine_deriv[2], floatTol) );
        CHECK_THAT( allderivs[3], WithinAbs(sine_deriv[3], floatTol) );
        CHECK_THAT( allderivs[4], WithinAbs(sine_deriv[4], floatTol) );
        CHECK     ( allderivs2[0] ==        sine_deriv2[0]             );
        CHECK     ( allderivs2[1] ==        sine_deriv2[1]             );
        CHECK_THAT( allderivs2[2], WithinAbs(sine_deriv2[2], floatTol) );
        CHECK_THAT( allderivs2[3], WithinAbs(sine_deriv2[3], floatTol) );
        CHECK_THAT( allderivs2[4], WithinAbs(sine_deriv2[4], floatTol) );
    }
}