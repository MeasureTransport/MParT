#include <catch2/catch_all.hpp>

#include "MParT/OrthogonalPolynomial.h"

#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing Probabilist Hermite normalization", "[ProbabilistHermiteNorm]" ) {

    ProbabilistHermite poly(false);

    // Gauss-Hermite quadrature points to integrate with exp(-0.5x^2) weight function
    Eigen::VectorXd quadPts(15);
    quadPts << -6.363947888829836,    
            -5.190093591304779,    
            -4.196207711269016,    
            -3.289082424398763,    
            -2.432436827009758,    
            -1.606710069028730,    
            -0.7991290683245480,    
            0,
            0.7991290683245480,    
            1.606710069028730,    
            2.432436827009756,    
            3.289082424398766,    
            4.196207711269015,    
            5.190093591304779,    
            6.363947888829833;  

   // Pulled from https://people.math.sc.edu/Burkardt/datasets/quadrature_rules_hermite_probabilist/hermite_probabilist_015_w.txt
   Eigen::VectorXd quadWts(15);
   quadWts << 0.2153105930760208E-08,
            0.1497815571693205E-05,
            0.1414276370885442E-03,
            0.3928782634853375E-02,
            0.4352954135285805E-01,
            0.2241371742044199,    
            0.5826965579477278,    
            0.7977583071397497,    
            0.5826965579477275,    
            0.2241371742044203,    
            0.4352954135285790E-01,
            0.3928782634853367E-02,
            0.1414276370885451E-03,
            0.1497815571693205E-05,
            0.2153105930760234E-08;


    Eigen::VectorXd evals(15);
    for(unsigned int order=0; order<10; ++order){
        for(unsigned int i=0; i<quadWts.size(); ++i)
            evals(i) = std::pow(poly.Evaluate(order, quadPts(i)), 2.0);
        
        double quadNorm = std::sqrt( (evals.array() * quadWts.array()).sum());
        double classNorm = poly.Normalization(order);

        CHECK(classNorm == Approx(quadNorm).epsilon(1e-7));
    }

}


TEST_CASE( "Testing Probabilist Hermite polynomials", "[ProbabilistHermite]" ) {

    const double floatTol = 1e-15;

    ProbabilistHermite poly;

    std::vector<double> xs{-1.0, -0.5, 0.0, 0.1, 1.0};
    std::vector<double> allvals(5);
    std::vector<double> allderivs(5);
    std::vector<double> allderivs2(5);
    
    // Check the evaluation
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

    // Check the derivative
    for(auto& x : xs){
        CHECK( poly.Derivative(0, x) == 0.0 ); 
        CHECK( poly.Derivative(1, x) == 1.0 );
        CHECK( poly.Derivative(2, x) == Approx(2.0*x).epsilon(floatTol) );
        CHECK( poly.Derivative(3, x) == Approx(3.0*x*x-3.0).epsilon(floatTol) );
        CHECK( poly.Derivative(4, x) == Approx(4.0*x*x*x - 12.0*x).epsilon(floatTol) );

        poly.EvaluateDerivatives(&allvals[0], &allderivs[0], 4, x);
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

        poly.EvaluateDerivatives(&allderivs[0], 4, x);
        CHECK( allderivs[0] == 0.0 ); 
        CHECK( allderivs[1] == 1.0 );
        CHECK( allderivs[2] == Approx(2.0*x).epsilon(floatTol) );
        CHECK( allderivs[3] == Approx(3.0*x*x-3.0).epsilon(floatTol) );
        CHECK( allderivs[4] == Approx(4.0*x*x*x - 12.0*x).epsilon(floatTol) );
    }

    // Check the second derivatives
    for(auto& x : xs){
        CHECK( poly.SecondDerivative(0, x) == 0.0 ); 
        CHECK( poly.SecondDerivative(1, x) == 0.0 );
        CHECK( poly.SecondDerivative(2, x) == Approx(2.0).epsilon(floatTol) );
        CHECK( poly.SecondDerivative(3, x) == Approx(6.0*x).epsilon(floatTol) );
        CHECK( poly.SecondDerivative(4, x) == Approx(12.0*x*x - 12.0).epsilon(floatTol) );

        poly.EvaluateSecondDerivatives(&allvals[0], &allderivs[0], &allderivs2[0], 4, x);
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

}



TEST_CASE( "Testing Physicist Hermite normalization", "[PhysicistHermiteNorm]" ) {

    PhysicistHermite poly(false);

    // Gauss-Hermite quadrature points to integrate with exp(-0.5x^2) weight function
    Eigen::VectorXd quadPts(15);
    quadPts <<   -4.499990707309390,    
                -3.669950373404451,    
                -2.967166927905604,    
                -2.325732486173856,    
                -1.719992575186489,    
                -1.136115585210921,    
                -0.5650695832555758,    
                0,
                0.5650695832555758,    
                1.136115585210921,    
                1.719992575186488,    
                2.325732486173858,    
                2.967166927905603,    
                3.669950373404451,    
                4.499990707309388; 

   // Pulled from https://people.math.sc.edu/Burkardt/datasets/quadrature_rules_hermite_physicist/hermite_physicist_015_w.txt
   Eigen::VectorXd quadWts(15);
   quadWts <<   0.1522475804253516E-08,
                0.1059115547711071E-05,
                0.1000044412324996E-03,
                0.2778068842912774E-02,
                0.3078003387254618E-01,
                0.1584889157959359,    
                0.4120286874988984,    
                0.5641003087264176,    
                0.4120286874988981,    
                0.1584889157959361,    
                0.3078003387254607E-01,
                0.2778068842912767E-02,
                0.1000044412325003E-03,
                0.1059115547711071E-05,
                0.1522475804253535E-08;


    Eigen::VectorXd evals(15);
    for(unsigned int order=0; order<10; ++order){
        for(unsigned int i=0; i<quadWts.size(); ++i)
            evals(i) = std::pow(poly.Evaluate(order, quadPts(i)), 2.0);
        
        double quadNorm = std::sqrt( (evals.array() * quadWts.array()).sum());
        double classNorm = poly.Normalization(order);

        CHECK(classNorm == Approx(quadNorm).epsilon(1e-7));
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


#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)

TEST_CASE( "Device Hermite polynomial evaluation", "[PhysicistHermiteDevice]" ) {

    const double floatTol = 1e-15;

    PhysicistHermite poly;

    std::vector<double> xs{-1.0, -0.5, 0.0, 0.1, 1.0};
    std::vector<double> allvals(5);
    
    Kokkos::View<double*,Kokkos::HostSpace> xs_host("host xs", 5);
    xs_host(0) = -1.0;
    xs_host(1) = -0.5;
    xs_host(2) = 0.0;
    xs_host(3) = 0.1;
    xs_host(4) = 1.0;

    auto xs_device = ToDevice<Kokkos::DefaultExecutionSpace::memory_space>(xs_host);

    Kokkos::View<double*,Kokkos::DefaultExecutionSpace::memory_space> ys_device("evals", xs.size());
    Kokkos::RangePolicy<typename Kokkos::DefaultExecutionSpace::execution_space> policy(0,xs.size());
    for(unsigned int p=0; p<10; ++p){
        Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const size_t ind){
            ys_device(ind) = poly.Evaluate(p, xs_device(ind));
        });
        
        auto ys_host = ToHost(ys_device);
        double trueVal;
        for(unsigned int i=0; i<xs.size(); ++i){
            trueVal = poly.Evaluate(p,xs_host(i));
            CHECK(ys_host(i) == Approx(trueVal).epsilon(floatTol));
        }
    }
}


#endif 
