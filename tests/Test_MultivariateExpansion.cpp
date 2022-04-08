#include <catch2/catch_all.hpp>

#include "MParT/MultivariateExpansion.h"
#include "MParT/OrthogonalPolynomial.h"

#include <Eigen/Dense>

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing multivariate expansion", "[MultivariateExpansion]") {

    const double testTol = 1e-7;

    unsigned int dim = 3;
    unsigned int maxDegree = 3; 
    FixedMultiIndexSet<Kokkos::HostSpace> mset(dim, maxDegree); // Create a total order limited fixed multindex set

    ProbabilistHermite poly1d;
    MultivariateExpansion<ProbabilistHermite,Kokkos::HostSpace> expansion(mset);

    unsigned int cacheSize = expansion.CacheSize();
    CHECK(cacheSize == (maxDegree+1)*(dim+2));

    // Allocate some memory for the cache 
    std::vector<double> cache(cacheSize);
    Eigen::VectorXd pt = Eigen::VectorXd::Random(dim);

    // Fill in the cache the first d-1 components of the cache  
    expansion.FillCache1(&cache[0], pt, DerivativeFlags::None);
    for(unsigned int d=0; d<dim-1;++d){
        for(unsigned int i=0; i<maxDegree+1; ++i){
            CHECK(cache[i + d*(maxDegree+1)] == Approx( poly1d.Evaluate(i,pt(d))).epsilon(1e-15) );
        }
    }

    // Fill in the last part of the cache for an evaluation
    expansion.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::None);
    for(unsigned int i=0; i<maxDegree+1; ++i){
        CHECK(cache[i + (dim-1)*(maxDegree+1)] == Approx( poly1d.Evaluate(i,pt(dim-1))).epsilon(1e-15) );
    }

    // Evaluate the expansion using the cache 
    Eigen::VectorXd coeffs = Eigen::VectorXd::Random(mset.Size());
    double f = expansion.Evaluate(&cache[0], coeffs);

    
    // Now fill in the last part of the cache for a gradient evaluation
    expansion.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::Diagonal);
    double df = expansion.DiagonalDerivative(&cache[0], coeffs,1);
    
    // Compare with a finite difference approximation of the derivative
    double fdStep = 1e-5;
    expansion.FillCache2(&cache[0], pt, pt(dim-1)+fdStep, DerivativeFlags::None);
    double f2 = expansion.Evaluate(&cache[0], coeffs);
    CHECK( df==Approx((f2-f)/fdStep).epsilon(1e-4));

    // Compute the second derivative
    expansion.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::Diagonal2);
    double d2f = expansion.DiagonalDerivative(&cache[0], coeffs,2);
    
    // Check with a finite difference second derivative 
    expansion.FillCache2(&cache[0], pt, pt(dim-1)+fdStep, DerivativeFlags::Diagonal);
    double df2 = expansion.DiagonalDerivative(&cache[0], coeffs,1);
    CHECK( d2f == Approx((df2-df)/fdStep).epsilon(1e-4));


    // Coefficient derivatives 
    expansion.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::Diagonal);
    
    Eigen::VectorXd grad(mset.Size());
    f2 = expansion.CoeffDerivative(&cache[0], coeffs, grad);
    CHECK(f2==Approx(f).epsilon(1e-15));

    // Check with a directional derivative in a random direction
    Eigen::VectorXd stepDir = Eigen::VectorXd::Random(mset.Size());
    stepDir /= stepDir.norm();

    Eigen::VectorXd coeffs2 = coeffs + fdStep * stepDir;
    f2 = expansion.Evaluate(&cache[0], coeffs2);

    CHECK( grad.dot(stepDir) == Approx((f2-f)/fdStep).epsilon(1e-4));



    // Mixed first derivatives
    df2 = expansion.MixedDerivative(&cache[0], coeffs, 1, grad);
    CHECK(df2==Approx(df).epsilon(1e-15));

    df2 = expansion.DiagonalDerivative(&cache[0], coeffs2, 1);
    CHECK( grad.dot(stepDir) == Approx((df2-df)/fdStep).epsilon(1e-4));

    
    // Mixed second derivatives (grad of d2f wrt coeffs)
    double d2f2 = expansion.MixedDerivative(&cache[0], coeffs, 2, grad);
    CHECK(d2f2==Approx(d2f).epsilon(1e-15));

    d2f2 = expansion.DiagonalDerivative(&cache[0], coeffs2, 2);
    CHECK( grad.dot(stepDir) == Approx((d2f2-d2f)/fdStep).epsilon(1e-4));
}