#include <catch2/catch_all.hpp>

#include "MParT/MonotoneComponent.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing monotone component integrand", "[MonotoneIntegrand]") {

    const double testTol = 1e-7;

    unsigned int dim = 1;
    unsigned int maxDegree = 1; 
    FixedMultiIndexSet mset(dim, maxDegree); // Create a total order limited fixed multindex set

    // Make room for the cache
    std::vector<double> cache((dim+2)*maxDegree);

    
    Kokkos::View<unsigned int*> startPos("Starting Positions", dim+3);
    startPos(0) = 0;
    for(unsigned int i=1; i<dim+1; ++i)
        startPos(i) = startPos(i-1) + maxDegree;
    startPos(dim+1) = startPos(dim) + maxDegree;
    startPos(dim+2) = startPos(dim+1) + maxDegree;

    Kokkos::View<double*> coeffs("Expansion coefficients", mset.Size());
    coeffs(0) = 1.0; // Constant term
    coeffs(1) = 1.0; // Linear term


    auto maxDegrees = mset.MaxDegrees();

    SECTION("Integral") {
        CachedMonotoneIntegrand<ProbabilistHermite, Exp> integrand(&cache[0], 1.0, startPos, maxDegrees, mset, coeffs, false);
        
        REQUIRE(integrand(0.0).size() == 1);
        CHECK(integrand(0.0)(0) == Approx(exp(1)).epsilon(testTol));
        CHECK(integrand(0.5)(0) == Approx(exp(1)).epsilon(testTol));
        CHECK(integrand(-0.5)(0) == Approx(exp(1)).epsilon(testTol));
    }

    SECTION("Integral Derivative") {
        CachedMonotoneIntegrand<ProbabilistHermite, Exp> integrand(&cache[0], 1.0, startPos, maxDegrees, mset, coeffs, true);
        
        REQUIRE(integrand(0.0).size() == 2);
        Eigen::Vector2d test = integrand(0.0);
        CHECK(test(0) == Approx(exp(1)).epsilon(testTol));

        test = integrand(0.5);
        CHECK(test(0) == Approx(exp(1)).epsilon(testTol));
        
        test = integrand(00.5);
        CHECK(test(0) == Approx(exp(1)).epsilon(testTol));
    }
}


TEST_CASE( "Testing monotone component evaluation in 1d", "[MonotoneComponent1d]" ) {

    const double testTol = 1e-7;
    unsigned int dim = 1;

    // Create points evently space on [lb,ub]
    unsigned int numPts = 20;
    double lb = -2.0;
    double ub = 2.0;

    Kokkos::View<double**> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        evalPts(0,i) = (i/double(numPts-1))*(ub-lb) - lb;
    
    /* Create and evaluate an affine map
       - Set coefficients so that f(x) = 1.0 + x
       - f(0) + int_0^x exp( d f(t) ) dt =  1.0 + int_0^x exp(1) dt = 1 + |x| * exp(1)
    */
    SECTION("Affine Map"){
        unsigned int maxDegree = 1; 
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
        
        
        Kokkos::View<double*> coeffs("Expansion coefficients", mset.Size());
        coeffs(0) = 1.0; // Constant term
        coeffs(1) = 1.0; // Linear term

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, absTol, relTol);

        MonotoneComponent<ProbabilistHermite, Exp, AdaptiveSimpson> comp(mset, quad);

        Kokkos::View<double*> output = comp.Evaluate(evalPts, coeffs);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK(output(i) == Approx(1+exp(1)*std::abs(evalPts(0,i))).epsilon(testTol));
        }
    }


    /* Create and evaluate a quadratic map
       - Set coefficients so that f(x) = 1.0 + x + 0.5*x^2
       - df/dt = 1.0 + t
       - f(0) + int_0^x exp( df/dt ) dt =  1.0 + int_0^x exp(1+t) dt = 1+exp(1+x)
    */
    SECTION("Quadratic Map"){
        unsigned int maxDegree = 2; 
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
        
        Kokkos::View<double*> coeffs("Expansion coefficients", mset.Size());
        coeffs(1) = 1.0; // Linear term = x ^1
        coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
        coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, absTol, relTol);

        MonotoneComponent<ProbabilistHermite, Exp, AdaptiveSimpson> comp(mset, quad);

        Kokkos::View<double*> output = comp.Evaluate(evalPts, coeffs);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK(output(i) == Approx(1+exp(1)*(exp(evalPts(0,i))-1)).epsilon(testTol));
        }
    }
}


TEST_CASE( "Testing monotone component derivative", "[MonotoneComponentDerivative]" ) {

    const double testTol = 1e-4;
    unsigned int dim = 2;
    const double fdStep = 1e-4;

    // Create points evently space on [lb,ub]
    unsigned int numPts = 20;
    double lb = -0.5;
    double ub = 0.5;

    Kokkos::View<double**> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        evalPts(0,i) = (i/double(numPts-1))*(ub-lb) - lb;
        evalPts(1,i) = evalPts(0,i);
    }
    
    Kokkos::View<double**> rightEvalPts("Finite difference points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        rightEvalPts(0,i) = evalPts(0,i);
        rightEvalPts(1,i) = evalPts(1,i) + fdStep;
    }
    
    unsigned int maxDegree = 2; 
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);

    unsigned int maxSub = 30;
    double relTol = 1e-7;
    double absTol = 1e-7;
    AdaptiveSimpson quad(maxSub, absTol, relTol);

    MonotoneComponent<ProbabilistHermite, Exp, AdaptiveSimpson> comp(mset, quad);
    
    // Create some arbitrary coefficients
    Kokkos::View<double*> coeffs("Expansion coefficients", mset.Size());
    for(unsigned int i=0; i<coeffs.extent(0); ++i)
        coeffs(i) = std::cos( 0.01*i );

    Kokkos::View<double*> evals = comp.Evaluate(evalPts, coeffs);
    Kokkos::View<double*> rightEvals = comp.Evaluate(rightEvalPts, coeffs);
    Kokkos::View<double*> contDerivs = comp.ContinuousDerivative(evalPts, coeffs);
    Kokkos::View<double*> discDerivs = comp.DiscreteDerivative(evalPts, coeffs);

    for(unsigned int i=0; i<numPts; ++i){
        double fdDeriv = (rightEvals(i)-evals(i))/fdStep;
        CHECK( contDerivs(i) == Approx(fdDeriv).epsilon(testTol) );
        CHECK( discDerivs(i) == Approx(fdDeriv).epsilon(testTol) );
    }
}