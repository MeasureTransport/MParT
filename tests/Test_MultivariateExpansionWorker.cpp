#include <catch2/catch_all.hpp>

#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/OrthogonalPolynomial.h"

#include "MParT/Utilities/ArrayConversions.h"

#include <Eigen/Dense>

using namespace mpart;
using namespace Catch;

TEST_CASE( "Testing multivariate expansion worker", "[MultivariateExpansionWorker]") {

    unsigned int dim = 3;
    unsigned int maxDegree = 3; 
    FixedMultiIndexSet<Kokkos::HostSpace> mset(dim, maxDegree); // Create a total order limited fixed multindex set

    ProbabilistHermite poly1d;
    MultivariateExpansionWorker<ProbabilistHermite,Kokkos::HostSpace> expansion(mset);

    unsigned int cacheSize = expansion.CacheSize();
    CHECK(cacheSize == (maxDegree+1)*(2*dim+1));

    // Allocate some memory for the cache 
    std::vector<double> cache(cacheSize);
    Kokkos::View<double*,Kokkos::HostSpace> pt("Point", dim);
    pt(0) = 0.2;
    pt(1) = 0.1;
    pt(2) = 0.345;
    
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
    Eigen::VectorXd coeffsEig = Eigen::VectorXd::Random(mset.Size());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> coeffs(coeffsEig.data(), coeffsEig.size());
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
    
    Eigen::VectorXd gradEig = -1.0*Eigen::VectorXd::Ones(mset.Size());
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> grad(gradEig.data(), gradEig.size());
    
    f2 = expansion.CoeffDerivative(&cache[0], coeffs, grad);
    CHECK(f2==Approx(f).epsilon(1e-15));

    // Check with a directional derivative in a random direction
    Eigen::VectorXd stepDir = Eigen::VectorXd::Random(mset.Size());
    stepDir /= stepDir.norm();

    Eigen::VectorXd coeffs2Eig = coeffsEig + fdStep * stepDir;
    Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> coeffs2(coeffs2Eig.data(), coeffs2Eig.size());
    
    f2 = expansion.Evaluate(&cache[0], coeffs2);

    CHECK( gradEig.dot(stepDir) == Approx((f2-f)/fdStep).epsilon(1e-4));

    // Mixed first derivatives
    df2 = expansion.MixedCoeffDerivative(&cache[0], coeffs, 1, grad);
    CHECK(df2==Approx(df).epsilon(1e-15));

    df2 = expansion.DiagonalDerivative(&cache[0], coeffs2, 1);
    CHECK( gradEig.dot(stepDir) == Approx((df2-df)/fdStep).epsilon(1e-4));

    
    // Mixed second derivatives (grad of d2f wrt coeffs)
    double d2f2 = expansion.MixedCoeffDerivative(&cache[0], coeffs, 2, grad);
    CHECK(d2f2==Approx(d2f).epsilon(1e-15));

    d2f2 = expansion.DiagonalDerivative(&cache[0], coeffs2, 2);
    CHECK( gradEig.dot(stepDir) == Approx((d2f2-d2f)/fdStep).epsilon(1e-4));

    SECTION("Input Derivatives"){
        // Check input derivatives 
        expansion.FillCache1(&cache[0], pt, DerivativeFlags::Input);
        expansion.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::Input);
            
        Kokkos::View<double*,Kokkos::HostSpace> inGrad("Input Gradient", dim);
        double eval = expansion.Evaluate(&cache[0], coeffs);
        double eval2 = expansion.InputDerivative(&cache[0], coeffs, inGrad);
        CHECK(eval2 == Approx(eval).epsilon(1e-13));

        for(unsigned int wrt=0; wrt<dim; ++wrt){
            pt(wrt) += fdStep;
            expansion.FillCache1(&cache[0], pt, DerivativeFlags::None);
            expansion.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::None);
            
            eval2 = expansion.Evaluate(&cache[0], coeffs);

            CHECK(inGrad(wrt) == Approx((eval2-eval)/fdStep).epsilon(1e-4));
            pt(wrt) -= fdStep;
        }    
    }

    SECTION("Mixed Input Derivatives"){
        // Check input derivatives 
        expansion.FillCache1(&cache[0], pt, DerivativeFlags::MixedInput);
        expansion.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::MixedInput);
            
        Kokkos::View<double*,Kokkos::HostSpace> inGrad("Input Gradient", dim);
        double df = expansion.DiagonalDerivative(&cache[0], coeffs, 1);
        double df2 = expansion.MixedInputDerivative(&cache[0], coeffs, inGrad);
        CHECK(df2 == Approx(df).epsilon(1e-13));

        for(unsigned int wrt=0; wrt<dim; ++wrt){
            pt(wrt) += fdStep;
            expansion.FillCache1(&cache[0], pt, DerivativeFlags::Diagonal);
            expansion.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::Diagonal);
            
            df2 = expansion.DiagonalDerivative(&cache[0], coeffs, 1);

            CHECK(inGrad(wrt) == Approx((df2-df)/fdStep).epsilon(1e-4));
            pt(wrt) -= fdStep;
        }    
    }
}


#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)

TEST_CASE( "Testing multivariate expansion on device", "[MultivariateExpansionWorkerDevice]") {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

    unsigned int dim = 3;
    unsigned int maxDegree = 3; 
    FixedMultiIndexSet<Kokkos::HostSpace> hset(dim,maxDegree);
    FixedMultiIndexSet<DeviceSpace> dset = hset.ToDevice<DeviceSpace>(); // Create a total order limited fixed multindex set

    MultivariateExpansionWorker<ProbabilistHermite,Kokkos::HostSpace> hexpansion(hset);
    MultivariateExpansionWorker<ProbabilistHermite,DeviceSpace> dexpansion(dset);

    unsigned int cacheSize = hexpansion.CacheSize();
    CHECK(cacheSize == (maxDegree+1)*(2*dim+1));

    // Allocate some memory for the cache 
    Kokkos::View<double*, Kokkos::HostSpace> hcache("host cache", cacheSize);
    Kokkos::View<double*, Kokkos::HostSpace> hcache2("host copy of device cache", cacheSize);
    
    Kokkos::View<double*, DeviceSpace> dcache("device cache", cacheSize);

    Kokkos::View<double*,Kokkos::HostSpace> hpt("host point", dim);
    for(unsigned int i=0; i<dim; ++i)
        hpt(i) = double(i)/dim;
    Kokkos::View<double*,DeviceSpace> dpt = ToDevice<DeviceSpace>(hpt);

    // Fill in the cache with the first d-1 components of the cache  
    hexpansion.FillCache1(hcache.data(), hpt, DerivativeFlags::None);
    
    // Run the fill cache funciton, using a parallel_for loop to ensure it's run on the device
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i){
        dexpansion.FillCache1(dcache.data(), dpt, DerivativeFlags::None);
        dexpansion.FillCache2(dcache.data(), dpt, 0.5 * dpt(dim-1), DerivativeFlags::None);
    });

    // Copy the device cache back to the host
    Kokkos::deep_copy(hcache2, dcache);

    // Check to make sure they're equal
    for(unsigned int d=0; d<dim-1;++d){
        for(unsigned int i=0; i<maxDegree+1; ++i){
            CHECK(hcache2[i + d*(maxDegree+1)] == Approx( hcache[i+d*(maxDegree+1)]).epsilon(1e-15) );
        }
    }
}

#endif 