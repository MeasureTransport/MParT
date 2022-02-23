#include <catch2/catch_all.hpp>

#include "MParT/MultivariateExpansion.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/TensorProductFunction.h"

#include <Eigen/Dense>

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing tensor product function", "[TensorProductFunction]") {

    unsigned int dim1 = 3;
    unsigned int dim2 = 2;
    unsigned int dim = dim1+dim2;

    unsigned int maxDegree = 3; 
    FixedMultiIndexSet mset1(dim1, maxDegree); // Create a total order limited fixed multindex set
    FixedMultiIndexSet mset2(dim2, maxDegree); // Create a total order limited fixed multindex set

    ProbabilistHermite poly1d;
    MultivariateExpansion<ProbabilistHermite> f1(mset1);
    MultivariateExpansion<ProbabilistHermite> f2(mset2);

    TensorProductFunction<decltype(f1), decltype(f2)> f(f1,f2);

    unsigned int cacheSize = f.CacheSize();
    CHECK(cacheSize == (f1.CacheSize() + f2.CacheSize()));

    // Allocate some memory for the cache 
    std::vector<double> cache(cacheSize);

    // Make up a point where we want to evaluate the tensor product function
    Kokkos::View<double*> pt("Evaluation point", dim);
    for(unsigned int i=0; i<dim; ++i)
        pt(i) = double(i+1)/dim;

    // Fill the cache with stuff that does not depend on x_d
    f.FillCache1(&cache[0], pt, DerivativeFlags::None);
    f.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::Diagonal2);

    Kokkos::View<double*> coeffs("joint coefficients", f.NumCoeffs());
    for(unsigned int i=0; i<f.NumCoeffs(); ++i)
        coeffs(i) = 0.1*i;

    auto coeffs1 = Kokkos::subview(coeffs, std::make_pair(int(0), int(dim1)));
    auto coeffs2 = Kokkos::subview(coeffs, std::make_pair(int(dim1),int(dim)));

    // Check evaluation
    double fval = f.Evaluate(&cache[0], coeffs1);

    double f1val = f1.Evaluate(&cache[0], coeffs1);
    double f2val = f2.Evaluate(&cache[f1.CacheSize()], coeffs2);
    double df2val = f2.DiagonalDerivative(&cache[f1.CacheSize()], coeffs2,1);
    double d2f2val = f2.DiagonalDerivative(&cache[f1.CacheSize()], coeffs2,2);

    CHECK( fval==Approx(f1val*f2val).epsilon(1e-15));


    // Check derivative
    f.FillCache1(&cache[0], pt, DerivativeFlags::None);
    f.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::Diagonal);

    double df = f.DiagonalDerivative(&cache[0], coeffs, 1);
    CHECK( df ==Approx(f1val*df2val).epsilon(1e-15));

    // Check second derivative
    double d2f = f.DiagonalDerivative(&cache[0], coeffs, 2);
    CHECK( d2f ==Approx(f1val*d2f2val).epsilon(1e-15));

    // Check coefficient gradient
    Eigen::VectorXd grad1(f1.NumCoeffs());
    Eigen::VectorXd grad2(f2.NumCoeffs());
    Eigen::VectorXd grad(f.NumCoeffs());

    f1.CoeffDerivative(&cache[0], coeffs1, grad1);
    f2.CoeffDerivative(&cache[f1.CacheSize()], coeffs2, grad2);

    double ftemp = f.CoeffDerivative(&cache[0], coeffs, grad);
    CHECK(ftemp==Approx(fval).epsilon(1e-15));

    for(unsigned int i=0; i<dim1; ++i)
        CHECK((f2val*grad1(i))==Approx(grad(i)).epsilon(1e-15));
    for(unsigned int i=0; i<dim2; ++i)
        CHECK((f1val*grad2(i))==Approx(grad(f1.NumCoeffs()+i)).epsilon(1e-15));  

    // Check the mixed gradients  
    f2.MixedDerivative(&cache[f1.CacheSize()], coeffs2, 1, grad2);
    f.MixedDerivative(&cache[0], coeffs, 1, grad);

    ftemp = f.MixedDerivative(&cache[0], coeffs, 1, grad);
    CHECK(ftemp==Approx(df).epsilon(1e-15));

    for(unsigned int i=0; i<dim1; ++i)
        CHECK((df2val*grad1(i))==Approx(grad(i)).epsilon(1e-15));
    for(unsigned int i=0; i<dim2; ++i)
        CHECK((f1val*grad2(i))==Approx(grad(f1.NumCoeffs()+i)).epsilon(1e-15));  
}

