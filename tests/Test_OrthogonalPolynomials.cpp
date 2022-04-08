#include <catch2/catch_all.hpp>

#include "MParT/OrthogonalPolynomial.h"

#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;
using namespace Catch;


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

    for(unsigned int p=0; p<10; ++p){
        Kokkos::parallel_for(xs.size(), KOKKOS_LAMBDA(const size_t ind){
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