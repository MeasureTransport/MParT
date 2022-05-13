#include <catch2/catch_all.hpp>

#include "MParT/MonotoneComponent.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansion.h"
#include "MParT/MonotoneIntegrand.h"

#include "MParT/Utilities/ArrayConversions.h"

#include <Eigen/Dense>

using namespace mpart;
using namespace Catch;

TEST_CASE( "MonotoneIntegrand1d", "[MonotoneIntegrand1d]") {

    const double testTol = 1e-7;

    unsigned int dim = 1;
    unsigned int maxDegree = 1; 
    FixedMultiIndexSet mset(dim, maxDegree); // Create a total order limited fixed multindex set

    MultivariateExpansion<ProbabilistHermite> expansion(mset);

    // // Make room for the cache
    std::vector<double> cache(expansion.CacheSize());
    REQUIRE(cache.size() == 6);

    Kokkos::View<double*, Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
    coeffs(0) = 1.0; // Constant term
    coeffs(1) = 1.0; // Linear term

    Kokkos::View<double*, Kokkos::HostSpace> pt("evaluation point", dim);
    pt(0) = 1.0;

    SECTION("Integrand Only") {
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*, Kokkos::HostSpace>, Kokkos::View<double*, Kokkos::HostSpace>> integrand(&cache[0], expansion, pt, coeffs, DerivativeFlags::None);
        
        REQUIRE(integrand(0.0).size() == 1);
        CHECK(integrand(0.0)(0) == Approx(exp(1)).epsilon(testTol));
        CHECK(integrand(0.5)(0) == Approx(exp(1)).epsilon(testTol));
        CHECK(integrand(-0.5)(0) == Approx(exp(1)).epsilon(testTol));
    }

    SECTION("Integrand Derivative") {
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*,Kokkos::HostSpace>, Kokkos::View<double*,Kokkos::HostSpace>> integrand(&cache[0], expansion, pt, coeffs, DerivativeFlags::Diagonal);
        
        REQUIRE(integrand(0.0).size() == 2);
        Eigen::Vector2d test = integrand(0.0);
        CHECK(test(0) == Approx(exp(1)).epsilon(testTol));

        test = integrand(0.5);
        CHECK(test(0) == Approx(exp(1)).epsilon(testTol));
        
        test = integrand(00.5);
        CHECK(test(0) == Approx(exp(1)).epsilon(testTol));
    }

    SECTION("Integrand Parameters Gradient") {
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*,Kokkos::HostSpace>,Kokkos::View<double*,Kokkos::HostSpace>> integrand(&cache[0], expansion, pt, coeffs, DerivativeFlags::Parameters);
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*,Kokkos::HostSpace>,Kokkos::View<double*,Kokkos::HostSpace>> integrand2(&cache[0], expansion, pt, coeffs, DerivativeFlags::None);
        
        Eigen::VectorXd testVal = integrand(0.5);
        REQUIRE(testVal.size() == 1+mset.Size());

        const double fdStep = 1e-4;
        for(unsigned int termInd=0; termInd<mset.Size(); ++termInd){
            coeffs(termInd) += fdStep;
            Eigen::VectorXd testVal2 = integrand2(0.5);
            double fdDeriv = (testVal2(0) - testVal(0))/fdStep;
            CHECK(testVal(termInd+1) == Approx(fdDeriv).epsilon(1e-4));
        }

    }


    SECTION("Integrand Mixed Gradient") {
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*,Kokkos::HostSpace>,Kokkos::View<double*,Kokkos::HostSpace>> integrand(&cache[0], expansion, pt, coeffs, DerivativeFlags::Mixed);
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*,Kokkos::HostSpace>,Kokkos::View<double*,Kokkos::HostSpace>> integrand2(&cache[0], expansion, pt, coeffs, DerivativeFlags::Diagonal);
        
        Eigen::VectorXd testVal = integrand(0.5);
        REQUIRE(testVal.size() == 1+mset.Size());

        const double fdStep = 1e-4;
        for(unsigned int termInd=0; termInd<mset.Size(); ++termInd){
            coeffs(termInd) += fdStep;
            Eigen::VectorXd testVal2 = integrand2(0.5);
            double fdDeriv = (testVal2(0) - testVal(0))/fdStep;
            CHECK(testVal(termInd+1) == Approx(fdDeriv).epsilon(1e-4));
        }
    }
}


TEST_CASE( "MonotoneIntegrand2d", "[MonotoneIntegrand2d]") {

    const double testTol = 1e-7;

    unsigned int dim = 2;
    unsigned int maxDegree = 3; 
    FixedMultiIndexSet mset(dim, maxDegree); // Create a total order limited fixed multindex set

    unsigned int numTerms = mset.Size();

    MultivariateExpansion<ProbabilistHermite> expansion(mset);

    // Make room for the cache
    std::vector<double> cache(expansion.CacheSize());
    
    Kokkos::View<double*,Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
    for(unsigned int i=0; i<coeffs.extent(0); ++i)
        coeffs(i) = 1.0/(i+1.0);
    
    Kokkos::View<double*,Kokkos::HostSpace> pt("evaluation point", dim);
    for(unsigned int i=0; i<dim; ++i)
        pt(i) = 0.25;

    // Set up the first part of the cache 
    expansion.FillCache1(&cache[0], pt, DerivativeFlags::None);

    SECTION("Integrand Only") {
        
        // Construct the integrand
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*,Kokkos::HostSpace>,Kokkos::View<double*,Kokkos::HostSpace>> integrand(&cache[0], expansion, pt, coeffs, DerivativeFlags::None);
        REQUIRE(integrand(0.0).size() == 1);

        // Evaluate the expansion 
        double df;
        for(double t : std::vector<double>{0.0, 0.5, -0.5, 1.0}){
            double xd = pt(dim-1);
            expansion.FillCache2(&cache[0], pt, t*xd, DerivativeFlags::Diagonal);
            df = expansion.DiagonalDerivative(&cache[0], coeffs, 1);
            CHECK(integrand(t)(0) == Approx(xd*exp(df)).epsilon(testTol));
        }
    }

    SECTION("Integrand Derivative") {
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*,Kokkos::HostSpace>,Kokkos::View<double*,Kokkos::HostSpace>> integrand(&cache[0], expansion, pt, coeffs, DerivativeFlags::Diagonal);
        REQUIRE(integrand(0.0).size() == 2);

        // Evaluate the expansion 
        double df, d2f;
        for(double t : std::vector<double>{0.0, 0.5, -0.5, 1.0}){
            double xd = pt(dim-1);
            expansion.FillCache2(&cache[0], pt, t*xd, DerivativeFlags::Diagonal2);
            df = expansion.DiagonalDerivative(&cache[0], coeffs, 1);
            d2f = expansion.DiagonalDerivative(&cache[0], coeffs, 2);

            CHECK(integrand(t)(0) == Approx(std::abs(xd)*exp(df)).epsilon(testTol));
            CHECK(integrand(t)(1) == Approx(exp(df) + xd*t*std::exp(df)*d2f).epsilon(testTol));   
        }
    }

    SECTION("Integrand Parameters Gradient") {
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*,Kokkos::HostSpace>,Kokkos::View<double*,Kokkos::HostSpace>> integrand(&cache[0], expansion, pt, coeffs, DerivativeFlags::Parameters);
        REQUIRE(integrand(0.0).size() == numTerms+1);

        // Evaluate the expansion 
        double df, d2f;
        Eigen::VectorXd coeffGrad(numTerms);

        for(double t : std::vector<double>{0.0, 0.5, -0.5, 1.0}){
            double xd = pt(dim-1);
            expansion.FillCache2(&cache[0], pt, t*xd, DerivativeFlags::Diagonal);
            df = expansion.DiagonalDerivative(&cache[0], coeffs, 1);
            expansion.MixedDerivative(&cache[0], coeffs, 1, coeffGrad);
            
            Eigen::VectorXd intVal = integrand(t);
            CHECK(intVal(0) == Approx(xd*exp(df)).epsilon(testTol));
            
            for(unsigned int i=0; i<numTerms; ++i)
                CHECK(intVal(i+1) == Approx(xd * std::exp(df)*coeffGrad(i)).epsilon(testTol));
            
        }
    }

    SECTION("Integrand Mixed Gradient") {
        MonotoneIntegrand<MultivariateExpansion<ProbabilistHermite>, Exp, Kokkos::View<double*,Kokkos::HostSpace>,Kokkos::View<double*,Kokkos::HostSpace>> integrand(&cache[0], expansion, pt, coeffs, DerivativeFlags::Mixed);
        REQUIRE(integrand(0.0).size() == numTerms+1);

        // Evaluate the expansion 
        double df, d2f;
        Eigen::VectorXd coeffGrad1(numTerms);
        Eigen::VectorXd coeffGrad2(numTerms);

        for(double t : std::vector<double>{0.0, 0.5, -0.5, 1.0}){
            double xd = pt(dim-1);

            expansion.FillCache2(&cache[0], pt, t*xd, DerivativeFlags::Diagonal2);
            df = expansion.DiagonalDerivative(&cache[0], coeffs, 1);
            d2f = expansion.DiagonalDerivative(&cache[0], coeffs, 2);
            expansion.MixedDerivative(&cache[0], coeffs, 1, coeffGrad1);
            double d2f2 = expansion.MixedDerivative(&cache[0], coeffs, 2, coeffGrad2);

            CHECK(d2f == Approx(d2f2).epsilon(1e-15));

            Eigen::VectorXd intVal = integrand(t);
            CHECK(intVal(0) == Approx(xd*exp(df)).epsilon(testTol));
            
            double dgdf = std::exp(df);
            double d2gdf = dgdf;

            for(unsigned int i=0; i<numTerms; ++i){
                double trueVal = (dgdf + xd * t * d2gdf * d2f) * coeffGrad1(i) + xd*t*dgdf*coeffGrad2(i);
                CHECK(intVal(i+1) == Approx(trueVal).epsilon(testTol));
            }
            
        }
    }
}



TEST_CASE( "Testing monotone component evaluation in 1d", "[MonotoneComponent1d]" ) {

    const double testTol = 1e-7;
    unsigned int dim = 1;

    // Create points evently space on [lb,ub]
    unsigned int numPts = 20;
    double lb = -2.0;
    double ub = 2.0;

    Kokkos::View<double**,Kokkos::HostSpace> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        evalPts(0,i) = (i/double(numPts-1))*(ub-lb) - lb;
    
    /* Create and evaluate an affine map
       - Set coefficients so that f(x) = 1.0 + x
       - f(0) + int_0^x exp( d f(t) ) dt =  1.0 + int_0^x exp(1) dt = 1 + |x| * exp(1)
    */
    SECTION("Affine Map"){
        unsigned int maxDegree = 1; 
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
        
        MultivariateExpansion<ProbabilistHermite> expansion(mset);
        
        Kokkos::View<double*,Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(0) = 1.0; // Constant term
        coeffs(1) = 1.0; // Linear term

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, absTol, relTol, QuadError::First);

        MonotoneComponent<MultivariateExpansion<ProbabilistHermite>, Exp, AdaptiveSimpson> comp(expansion, quad);

        Kokkos::View<double*,Kokkos::HostSpace> output = comp.EvaluateImpl(evalPts, coeffs);

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
       
        MultivariateExpansion<ProbabilistHermite> expansion(mset);
        
        Kokkos::View<double*,Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(1) = 1.0; // Linear term = x ^1
        coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
        coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, absTol, relTol,QuadError::First);

        MonotoneComponent<MultivariateExpansion<ProbabilistHermite>, Exp, AdaptiveSimpson> comp(expansion, quad);

        Kokkos::View<double*,Kokkos::HostSpace> output = comp.EvaluateImpl(evalPts, coeffs);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK(output(i) == Approx(1+exp(1)*(exp(evalPts(0,i))-1)).epsilon(testTol));
        }
    }
}


TEST_CASE( "Testing bracket-based inversion of monotone component", "[MonotoneBracketInverse]" ) {

    const double testTol = 1e-7;
    unsigned int dim = 1;

    // Create points evently space on [lb,ub]
    unsigned int numPts = 20;
    double lb = -2.0;
    double ub = 2.0;

    Kokkos::View<double**, Kokkos::HostSpace> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        evalPts(0,i) = (i/double(numPts-1))*(ub-lb) - lb;
    
    /* Create and evaluate an affine map
       - Set coefficients so that f(x) = 1.0 + x
       - f(0) + int_0^x exp( d f(t) ) dt =  1.0 + int_0^x exp(1) dt = 1 + |x| * exp(1)
    */
    SECTION("Affine Map"){
        unsigned int maxDegree = 1; 
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
        
        MultivariateExpansion<ProbabilistHermite> expansion(mset);
        
        Kokkos::View<double*, Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(0) = 1.0; // Constant term
        coeffs(1) = 1.0; // Linear term

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, absTol, relTol, QuadError::First);

        MonotoneComponent<MultivariateExpansion<ProbabilistHermite>, Exp, AdaptiveSimpson> comp(expansion, quad);

        Kokkos::View<double*, Kokkos::HostSpace> ys = comp.EvaluateImpl(evalPts, coeffs);
        Kokkos::View<double*, Kokkos::HostSpace> testInverse("Test output", numPts);

        comp.InverseImpl(evalPts, ys, coeffs, testInverse);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK(testInverse(i) == Approx(evalPts(0,i)).epsilon(testTol));
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
        
        MultivariateExpansion<ProbabilistHermite> expansion(mset);
        
        Kokkos::View<double*, Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(1) = 1.0; // Linear term = x ^1
        coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
        coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, absTol, relTol,QuadError::First);

        MonotoneComponent<MultivariateExpansion<ProbabilistHermite>, Exp, AdaptiveSimpson> comp(expansion, quad);

        Kokkos::View<double*, Kokkos::HostSpace> ys = comp.EvaluateImpl(evalPts, coeffs);
        Kokkos::View<double*, Kokkos::HostSpace> testInverse = comp.InverseImpl(evalPts, ys, coeffs);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK(testInverse(i) == Approx(evalPts(0,i)).epsilon(testTol));
        }
    }


    SECTION("Same x, multiple ys"){

        Kokkos::View<double**, Kokkos::HostSpace> x("Evaluate Points", dim, 1);
        x(0,0) = 0.5*(lb+ub);
        
        unsigned int maxDegree = 2; 
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
        MultivariateExpansion<ProbabilistHermite> expansion(mset);
        
        Kokkos::View<double*, Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(1) = 1.0; // Linear term = x ^1
        coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
        coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, absTol, relTol,QuadError::First);

        MonotoneComponent<MultivariateExpansion<ProbabilistHermite>, Exp, AdaptiveSimpson> comp(expansion, quad);

        Kokkos::View<double*, Kokkos::HostSpace> ys = comp.EvaluateImpl(evalPts, coeffs);
        Kokkos::View<double*, Kokkos::HostSpace> testInverse = comp.InverseImpl(x, ys, coeffs);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK(testInverse(i) == Approx(evalPts(0,i)).epsilon(1e-6));
        }
    }
}



TEST_CASE( "Testing monotone component derivative", "[MonotoneComponentDerivative]" ) {

    const double testTol = 1e-4;
    unsigned int dim = 2;
    const double fdStep = 1e-4;

    // Create points evently spaced on [lb,ub]
    unsigned int numPts = 20;
    double lb = -0.5;
    double ub = 0.5;

    Kokkos::View<double**, Kokkos::HostSpace> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        evalPts(0,i) = (i/double(numPts-1))*(ub-lb) - lb;
        evalPts(1,i) = evalPts(0,i);
    }
    
    Kokkos::View<double**, Kokkos::HostSpace> rightEvalPts("Finite difference points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        rightEvalPts(0,i) = evalPts(0,i);
        rightEvalPts(1,i) = evalPts(1,i) + fdStep;
    }
    
    unsigned int maxDegree = 2; 
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
    MultivariateExpansion<ProbabilistHermite> expansion(mset);
        

    unsigned int numTerms = mset.Size();

    unsigned int maxSub = 30;
    double relTol = 1e-7;
    double absTol = 1e-7;
    AdaptiveSimpson quad(maxSub, absTol, relTol,QuadError::First);

    MonotoneComponent<MultivariateExpansion<ProbabilistHermite>, Exp, AdaptiveSimpson> comp(expansion, quad);
    
    // Create some arbitrary coefficients
    Kokkos::View<double*, Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
    for(unsigned int i=0; i<coeffs.extent(0); ++i)
        coeffs(i) = 0.1*std::cos( 0.01*i );

    Kokkos::View<double*, Kokkos::HostSpace> evals = comp.EvaluateImpl(evalPts, coeffs);
    Kokkos::View<double*, Kokkos::HostSpace> rightEvals = comp.EvaluateImpl(rightEvalPts, coeffs);
    Kokkos::View<double*, Kokkos::HostSpace> contDerivs = comp.ContinuousDerivative(evalPts, coeffs);
    Kokkos::View<double*, Kokkos::HostSpace> discDerivs = comp.DiscreteDerivative(evalPts, coeffs);

    for(unsigned int i=0; i<numPts; ++i){
        double fdDeriv = (rightEvals(i)-evals(i))/fdStep;
        CHECK( contDerivs(i) == Approx(fdDeriv).epsilon(testTol) );
        CHECK( discDerivs(i) == Approx(fdDeriv).epsilon(testTol) );
    }

    SECTION("Coefficient Jacobian"){

        Kokkos::View<double*, Kokkos::HostSpace> evals2("FD Evals", numPts);
        Kokkos::View<double**, Kokkos::HostSpace> jac("Jacobian", numPts, numTerms);

        comp.CoeffJacobian(evalPts, coeffs, evals2, jac);

        for(unsigned int i=0; i<numPts; ++i)    
            CHECK(evals2(i) == Approx(evals(i)).epsilon(1e-12));

        const double fdStep = 1e-4;

        for(unsigned j=0; j<numTerms; ++j){
            coeffs(j) += fdStep;
            evals2 = comp.EvaluateImpl(evalPts, coeffs);

            for(unsigned int i=0; i<numPts; ++i) 
                CHECK(jac(i,j) == Approx((evals2(i)-evals(i))/fdStep).epsilon(5e-4).margin(1e-4));

            coeffs(j) -= fdStep;
        }
    }


     SECTION("Mixed Discrete Jacobian"){

        const double fdStep = 1e-4;

        Kokkos::View<double*, Kokkos::HostSpace> derivs("Derivatives", numPts);
        Kokkos::View<double*, Kokkos::HostSpace> derivs2("Derivatives2", numPts);
        
        Kokkos::View<double**, Kokkos::HostSpace> jac("Jacobian", numPts, numTerms);

        comp.DiscreteMixedJacobian(evalPts, coeffs, jac);
        derivs = comp.DiscreteDerivative(evalPts, coeffs);

        // Perturb the coefficients and recompute
        Kokkos::View<double*, Kokkos::HostSpace> coeffs2("Coefficients2", numTerms);
        Kokkos::deep_copy(coeffs2, coeffs);

        for(unsigned int j=0; j<coeffs.extent(0); ++j){
            coeffs2(j) += fdStep;
            derivs2 = comp.DiscreteDerivative(evalPts, coeffs2);

            for(unsigned int i=0; i<derivs2.extent(0); ++i){
                CHECK(jac(i,j)==Approx((derivs2(i) - derivs(i))/fdStep).epsilon(1e-4).margin(1e-3));
            }

            coeffs2(j) = coeffs(j);
        }

    }

    SECTION("Mixed Continuous Jacobian"){

        const double fdStep = 1e-4;

        Kokkos::View<double*, Kokkos::HostSpace> derivs("Derivatives", numPts);
        Kokkos::View<double*, Kokkos::HostSpace> derivs2("Derivatives2", numPts);
        
        Kokkos::View<double**, Kokkos::HostSpace> jac("Jacobian", numPts, numTerms);

        comp.ContinuousMixedJacobian(evalPts, coeffs, jac);
        derivs = comp.ContinuousDerivative(evalPts, coeffs);

        // Perturb the coefficients and recompute
        Kokkos::View<double*, Kokkos::HostSpace> coeffs2("Coefficients2", numTerms);
        Kokkos::deep_copy(coeffs2, coeffs);

        for(unsigned int j=0; j<coeffs.extent(0); ++j){
            coeffs2(j) += fdStep;
            derivs2 = comp.ContinuousDerivative(evalPts, coeffs2);

            for(unsigned int i=0; i<derivs2.extent(0); ++i){
                CHECK(jac(i,j)==Approx((derivs2(i) - derivs(i))/fdStep).epsilon(1e-4).margin(1e-3));
            }

            coeffs2(j) = coeffs(j);
        }

    }
}



TEST_CASE( "Least squares test", "[MonotoneComponentRegression]" ) {
    
    unsigned int numPts = 100;
    Kokkos::View<double**,Kokkos::HostSpace> pts("Training Points", 1,numPts);
    for(unsigned int i=0; i<numPts; ++i)
        pts(0,i) = i/(numPts-1.0);
    

    Kokkos::View<double*,Kokkos::HostSpace> fvals("Training Vales", numPts);
    for(unsigned int i=0; i<numPts; ++i)
        fvals(i) = pts(0,i)*pts(0,i) + pts(0,i);

    
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(1, 6);
    MultivariateExpansion<ProbabilistHermite> expansion(mset);
   
    unsigned int maxSub = 30;
    double relTol = 1e-3;
    double absTol = 1e-3;
    AdaptiveSimpson quad(maxSub, absTol, relTol, QuadError::First);

    MonotoneComponent<MultivariateExpansion<ProbabilistHermite>, SoftPlus, AdaptiveSimpson> comp(expansion, quad);

    unsigned int numTerms = mset.Size();
    Kokkos::View<double*,Kokkos::HostSpace> coeffs("Coefficients", numTerms);
    Kokkos::View<double**,Kokkos::HostSpace> jac("Gradient", numPts, numTerms);
    Kokkos::View<double*,Kokkos::HostSpace> preds("Predictions", numPts);


    double objective;

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>> jacMat(&jac(0,0), numPts, numTerms);

    Eigen::Map<Eigen::VectorXd> predVec(&preds(0), numPts);
    Eigen::Map<Eigen::VectorXd> obsVec(&fvals(0), numPts);
    Eigen::Map<Eigen::VectorXd> coeffVec(&coeffs(0), numTerms);

    Eigen::VectorXd objGrad;

    for(unsigned int optIt=0; optIt<5; ++optIt){

        comp.CoeffJacobian(pts, coeffs, preds, jac);
        
        objGrad = predVec-obsVec;
        
        objective = 0.5*objGrad.squaredNorm();
        coeffVec -= jacMat.colPivHouseholderQr().solve(objGrad);
    }
    
    CHECK(objective<1e-3);

}


#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)

TEST_CASE( "MonotoneIntegrand1d on device", "[MonotoneIntegrandDevice]") {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

    const double testTol = 1e-7;

    unsigned int dim = 1;
    unsigned int maxDegree = 1; 
    FixedMultiIndexSet<Kokkos::HostSpace> hset(dim, maxDegree);
    FixedMultiIndexSet<DeviceSpace> mset = hset.ToDevice(); // Create a total order limited fixed multindex set

    MultivariateExpansion<ProbabilistHermite,DeviceSpace> expansion(mset);

    MultivariateExpansion<ProbabilistHermite,Kokkos::HostSpace> hexpansion(hset);

    // Make room for the cache
    unsigned int cacheSize = hexpansion.CacheSize();
    Kokkos::View<double*, DeviceSpace> dcache("device cache", cacheSize);

    Kokkos::View<double*, Kokkos::HostSpace> hcoeffs("Expansion coefficients", mset.Size());
    hcoeffs(0) = 1.0; // Constant term
    hcoeffs(1) = 1.0; // Linear term

    Kokkos::View<double*, DeviceSpace> dcoeffs = ToDevice<DeviceSpace>(hcoeffs);

    Kokkos::View<double*, Kokkos::HostSpace> hpt("evaluation point", dim);
    hpt(0) = 1.0;

    Kokkos::View<double*, DeviceSpace> dpt = ToDevice<DeviceSpace>(hpt);

    SECTION("Integrand Only") {
        Kokkos::View<double*, DeviceSpace> dres("result", 1);

        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i){
            MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs)> integrand(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::None);
            dres(0) = integrand(0.5)(0);
        });
        
        Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);
        
        CHECK(hres(0) == Approx(exp(1)).epsilon(testTol));
    }

    SECTION("Integrand Derivative") {
        Kokkos::View<double*, DeviceSpace> dres("result", 2);

        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i){
            MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs)> integrand(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::Diagonal);
        
            Eigen::VectorXd result = integrand(0.5);
            dres(0) = result(0);
            dres(1) = result(1);
        });

        Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);

        CHECK(hres(0) == Approx(exp(1)).epsilon(testTol));
    }

    SECTION("Integrand Parameters Gradient") {
         
        Kokkos::View<double*, DeviceSpace> dres("result", hset.Size());
        Kokkos::View<double*, DeviceSpace> dres_fd("result_fd", hset.Size());

        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i){

            MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs)> integrand(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::Parameters);
            MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs)> integrand2(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::None);
      
            Eigen::VectorXd testVal = integrand(0.5);
            
            const double fdStep = 1e-4;
            for(unsigned int termInd=0; termInd<dres.extent(0); ++termInd){
                dcoeffs(termInd) += fdStep;
                Eigen::VectorXd testVal2 = integrand2(0.5);
                dres_fd(termInd) = (testVal2(0) - testVal(0))/fdStep;
                dres(termInd) = testVal(1 + termInd);
            }
            
        });

        Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);
        Kokkos::View<double*, Kokkos::HostSpace> hres_fd = ToHost(dres_fd);
        
        for(unsigned int termInd=0; termInd<hres.extent(0); ++termInd)
            CHECK(hres(termInd) == Approx(hres_fd(termInd)).epsilon(1e-4));
    }


    SECTION("Integrand Mixed Gradient") {

        Kokkos::View<double*, DeviceSpace> dres("result", hset.Size());
        Kokkos::View<double*, DeviceSpace> dres_fd("result_fd", hset.Size());

        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i){

            MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs)> integrand(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::Mixed);
            MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs)> integrand2(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::Diagonal);
            
            Eigen::VectorXd testVal = integrand(0.5);

            const double fdStep = 1e-4;
            for(unsigned int termInd=0; termInd<dres.extent(0); ++termInd){
                dcoeffs(termInd) += fdStep;
                Eigen::VectorXd testVal2 = integrand2(0.5);
                dres_fd(termInd) = (testVal2(0) - testVal(0))/fdStep;
                dres(termInd) = testVal(1 + termInd);
            }
        });

        Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);
        Kokkos::View<double*, Kokkos::HostSpace> hres_fd = ToHost(dres_fd);
        
        for(unsigned int termInd=0; termInd<hres.extent(0); ++termInd)
            CHECK(hres(termInd) == Approx(hres_fd(termInd)).epsilon(1e-4));

    }
}



TEST_CASE( "Testing MonotoneComponent::EvaluateSingle on Device", "[MonotoneComponentSingle_Device]") {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

    const double testTol = 1e-7;

    unsigned int dim = 2;
    unsigned int maxDegree = 1; 
    FixedMultiIndexSet<Kokkos::HostSpace> hset(dim,maxDegree);
    FixedMultiIndexSet<DeviceSpace> dset = hset.ToDevice(); // Create a total order limited fixed multindex set

    MultivariateExpansion<ProbabilistHermite,DeviceSpace> dexpansion(dset);

    // define f(x1,x2) = c0 + c1*x1 + c2*x2
    Kokkos::View<double*,Kokkos::HostSpace> hcoeffs("Expansion coefficients", hset.Size());
    hcoeffs(0) = 1.0; // Constant term
    hcoeffs(1) = 1.0; // Linear term in x1
    hcoeffs(2) = 1.0; // Linear in x2
    
    Kokkos::View<double*,DeviceSpace> dcoeffs = ToDevice<DeviceSpace>(hcoeffs);
    
    unsigned int cacheSize = dexpansion.CacheSize();
    CHECK(cacheSize == (maxDegree+1)*(dim+2));

    // Allocate some memory for the cache 
    Kokkos::View<double*, DeviceSpace> dcache("device cache", cacheSize);

    Kokkos::View<double*,Kokkos::HostSpace> hpt("host point", dim);
    hpt(0) = 0.5;
    hpt(1) = 0.5;
    Kokkos::View<double*,DeviceSpace> dpt = ToDevice<DeviceSpace>(hpt);

    unsigned int maxSub = 20;
    double relTol = 1e-5;
    double absTol = 1e-5;
    AdaptiveSimpson quad(maxSub, absTol, relTol, QuadError::First);
    
    Kokkos::View<double*, DeviceSpace> dres("Device Evaluation", 1);
    // Run the fill cache funciton, using a parallel_for loop to ensure it's run on the device
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i){
        dexpansion.FillCache1(dcache.data(), dpt, DerivativeFlags::None);
        dres(0) = MonotoneComponent<decltype(dexpansion),Exp, decltype(quad)>::EvaluateSingle(dcache.data(), dpt, dpt(dim-1), dcoeffs, quad, dexpansion);
    });

    Kokkos::fence();

    CHECK(ToHost(dres)(0) == Approx(hcoeffs(0) + hcoeffs(1)*hpt(0) + hpt(1)*exp(hcoeffs(2))).epsilon(1e-4));

}


TEST_CASE( "Testing 1d monotone component evaluation on device", "[MonotoneComponent1d_Device]" ) {

    std::cout << "Entering test..." << std::endl;

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;
    
    const double testTol = 1e-7;
    unsigned int dim = 1;

    // Create points evently space on [lb,ub]
    unsigned int numPts = 1;
    double lb = -2.0;
    double ub = 2.0;

    Kokkos::View<double**,DeviceSpace>::HostMirror hevalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        hevalPts(0,i) = (i/double(numPts-1))*(ub-lb) - lb;

    Kokkos::View<double**,DeviceSpace> devalPts = ToDevice<DeviceSpace>(hevalPts);


    /* Create and evaluate an affine map
       - Set coefficients so that f(x) = 1.0 + x
       - f(0) + int_0^x exp( d f(t) ) dt =  1.0 + int_0^x exp(1) dt = 1 + |x| * exp(1)
    */
    SECTION("Affine Map"){
        unsigned int maxDegree = 1; 
        FixedMultiIndexSet<Kokkos::HostSpace> hset(dim, maxDegree);
        FixedMultiIndexSet<DeviceSpace> mset = hset.ToDevice();
        
        MultivariateExpansion<ProbabilistHermite,DeviceSpace> expansion(mset);
        
        Kokkos::View<double*,Kokkos::HostSpace> hcoeffs("Expansion coefficients", mset.Size());
        hcoeffs(0) = 1.0; // Constant term
        hcoeffs(1) = 1.0; // Linear term
        
        Kokkos::View<double*,DeviceSpace> dcoeffs = ToDevice<DeviceSpace>(hcoeffs);

        unsigned int maxSub = 1;
        double relTol = 1e-3;
        double absTol = 1e-3;
        AdaptiveSimpson quad(maxSub, absTol, relTol, QuadError::First);

        MonotoneComponent<decltype(expansion), Exp, AdaptiveSimpson> comp(expansion, quad);

        Kokkos::View<double*,DeviceSpace> output = comp.EvaluateImpl<DeviceSpace,Kokkos::Cuda>(devalPts, dcoeffs);

    //     // for(unsigned int i=0; i<numPts; ++i){
    //     //     CHECK(output(i) == Approx(1+exp(1)*std::abs(evalPts(0,i))).epsilon(testTol));
    //     // }
    }


    // /* Create and evaluate a quadratic map
    //    - Set coefficients so that f(x) = 1.0 + x + 0.5*x^2
    //    - df/dt = 1.0 + t
    //    - f(0) + int_0^x exp( df/dt ) dt =  1.0 + int_0^x exp(1+t) dt = 1+exp(1+x)
    // */
    // SECTION("Quadratic Map"){
    //     unsigned int maxDegree = 2; 
    //     MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
    //     ProbabilistHermite poly1d;
    //     MultivariateExpansion<ProbabilistHermite> expansion(mset);
        
    //     Kokkos::View<double*,Kokkos::HostSpace> coeffs("Expansion coefficients", mset.Size());
    //     coeffs(1) = 1.0; // Linear term = x ^1
    //     coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
    //     coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

    //     unsigned int maxSub = 30;
    //     double relTol = 1e-7;
    //     double absTol = 1e-7;
    //     AdaptiveSimpson quad(maxSub, absTol, relTol,QuadError::First);

    //     MonotoneComponent<MultivariateExpansion<ProbabilistHermite>, Exp, AdaptiveSimpson> comp(expansion, quad);

    //     Kokkos::View<double*,Kokkos::HostSpace> output = comp.Evaluate(evalPts, coeffs);

    //     for(unsigned int i=0; i<numPts; ++i){
    //         CHECK(output(i) == Approx(1+exp(1)*(exp(evalPts(0,i))-1)).epsilon(testTol));
    //     }
    // }

    std::cout << "Leaving test..." << std::endl;
}

#endif