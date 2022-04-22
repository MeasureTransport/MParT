#include <catch2/catch_all.hpp>

#include "MParT/Quadrature.h"
#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;
using namespace Catch;


class TestIntegrand {
public:

    KOKKOS_INLINE_FUNCTION double operator()(double x) const{
        return exp(x);
    }

}; // class TestIntegrand

TEST_CASE( "Testing CC Quadrature", "[ClenshawCurtisQuadrature]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int order = 10;

    // Set tolerance for tests
    double testTol = 1e-8;

    ClenshawCurtisQuadrature quad(order);

    SECTION("Class Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        TestIntegrand integrand;
        double integral = quad.Integrate<double>(integrand, lb, ub);

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

    SECTION("Lambda Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        auto integrand = [](double x){return exp(x);};
        double integral = quad.Integrate<double>(integrand, lb, ub);    

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

    SECTION("Vector-Valued Integrand")
    {
        double lb = 0.0;
        double ub = 1.0;

        auto integrand = [](double x)->Eigen::VectorXd {return exp(x)*Eigen::VectorXd::Ones(2).eval();};

        auto integral = quad.Integrate<Eigen::VectorXd>(integrand, lb, ub);    

        REQUIRE(integral.size()==2);
        CHECK( integral(0) == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( integral(1) == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

}


TEST_CASE( "Testing Recursive Quadrature", "[RecursiveQuadrature]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int maxSub = 10;
    double relTol = 1e-7;
    double absTol = 1e-7;
    unsigned int order = 8;

    // Set tolerance for tests
    double testTol = 1e-4;

    AdaptiveClenshawCurtis quad(maxSub, absTol, relTol,QuadError::First, order);

    SECTION("Class Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        TestIntegrand integrand;
        double integral = quad.Integrate<double>(integrand, lb, ub);

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        //CHECK( quad.Status()>0 );
        //CHECK( quad.MaxLevel()<maxSub );
    }

    SECTION("Lambda Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        auto integrand = [](double x){return exp(x);};
        double integral = quad.Integrate<double>(integrand, lb, ub);    

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        //CHECK( quad.Status()>0 );
        //CHECK( quad.MaxLevel()<maxSub );
    }


    SECTION("Discontinuous Integrand")
    {   
        double lb = 0;
        double ub = 1.0;
        unsigned int numEvals = 0;

        auto integrand = [&](double x){
            numEvals++;
            if(x<0.5)
                return exp(x);
            else 
                return 1.0+exp(x);
        };
        double integral = quad.Integrate<double>(integrand, lb, ub);    

        double trueVal = (ub-0.5) + exp(ub)-exp(lb);
        CHECK( integral == Approx(trueVal).epsilon(testTol) );
        //CHECK( quad.Status()>0 );
        //CHECK( quad.MaxLevel()<=maxSub );
        CHECK( numEvals<400);
    }
    
    SECTION("Vector-Valued Integrand")
    {
        double lb = 0.0;
        double ub = 1.0;

        auto integrand = [](double x)->Eigen::VectorXd {return exp(x)*Eigen::VectorXd::Ones(2).eval();};

        auto integral = quad.Integrate<Eigen::VectorXd>(integrand, lb, ub);    

        REQUIRE(integral.size()==2);
        CHECK( integral(0) == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( integral(1) == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }
}



TEST_CASE( "Testing Adaptive Simpson Integration", "[AdaptiveSimpson]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int maxSub = 30;
    double relTol = 1e-7;
    double absTol = 1e-7;

    // Set tolerance for tests
    double testTol = 1e-4;

    AdaptiveSimpson quad(maxSub, absTol, relTol, QuadError::First);

    SECTION("Class Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        TestIntegrand integrand;
        double integral = quad.Integrate<double>(integrand, lb, ub);

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        // CHECK( quad.Status()>0 );
        // CHECK( quad.MaxLevel()<maxSub );
    }

    SECTION("Lambda Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        auto integrand = [](double x){return exp(x);};
        double integral = quad.Integrate<double>(integrand, lb, ub);    

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        // CHECK( quad.Status()>0 );
        // CHECK( quad.MaxLevel()<maxSub );
    }


    SECTION("Discontinuous Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        unsigned int numEvals = 0;

        auto integrand = [&](double x){
            numEvals++;
            if(x<0.5)
                return exp(x);
            else 
                return 1.0+exp(x);
        };
        double integral = quad.Integrate<double>(integrand, lb, ub);    

        double trueVal = (ub-0.5) + exp(ub)-exp(lb);
        CHECK( integral == Approx(trueVal).epsilon(testTol) );
        // CHECK( quad.Status()>0 );
        // CHECK( quad.MaxLevel()<=maxSub );
        CHECK( numEvals<150);
    }

    SECTION("Vector-Valued Integrand")
    {
        double lb = 0.0;
        double ub = 1.0;

        auto integrand = [](double x)->Eigen::VectorXd {return exp(x)*Eigen::VectorXd::Ones(2).eval();};

        auto integral = quad.Integrate<Eigen::VectorXd>(integrand, lb, ub);    

        REQUIRE(integral.size()==2);
        CHECK( integral(0) == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( integral(1) == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }
}




#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)


TEST_CASE( "Testing CC Quadrature on device", "[ClenshawCurtisDevice]" ) {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

    // Set parameters for adaptive quadrature algorithm
    unsigned int order = 10;
    unsigned int numRepeats = 5;

    // Set tolerance for tests
    double testTol = 1e-8;
    double lb = 0;
    double ub = 1.0;

    TestIntegrand integrand;
    ClenshawCurtisQuadrature quad(order);
    
    // TestIntegrand integrand;
    Kokkos::View<double*, DeviceSpace> dres("integrals", numRepeats);
    Kokkos::parallel_for(numRepeats, KOKKOS_LAMBDA(const unsigned int i){
        dres(i) = quad.Integrate<double>(integrand, lb, ub);
    });

    Kokkos::fence();
    Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);
    double integral = quad.Integrate<double>(integrand, lb, ub);

    for(unsigned int i=0; i<numRepeats; ++i)
        CHECK(hres(i) == Approx(integral).epsilon(1e-7));

}



TEST_CASE( "Testing Adaptive Simpson Quadrature on device", "[AdaptiveSimpsonDevice]" ) {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

    unsigned int numRepeats = 5;

    // Set parameters for adaptive quadrature algorithm
    unsigned int maxSub = 30;
    double relTol = 1e-6;
    double absTol = 1e-6;
    
    // Set tolerance for tests
    double testTol = 1e-4;
    double lb = 0;
    double ub = 1.0;

    TestIntegrand integrand;
    AdaptiveSimpson quad(maxSub, absTol, relTol, QuadError::First);

    
    // TestIntegrand integrand;
    Kokkos::View<double*, DeviceSpace> dres("integrals", numRepeats);
    Kokkos::parallel_for(numRepeats, KOKKOS_LAMBDA(const unsigned int i){
        dres(i) = quad.Integrate<double>(integrand, lb, ub);
    });

    Kokkos::fence();
    Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);
    double integral = quad.Integrate<double>(integrand, lb, ub);

    for(unsigned int i=0; i<numRepeats; ++i)
        CHECK(hres(i) == Approx(integral).epsilon(1e-3));

}


TEST_CASE( "Testing Adaptive Clenshaw Curtis on device", "[AdaptiveCCDevice]" ) {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

    unsigned int numRepeats = 5;

    unsigned int maxSub = 10;
    double relTol = 1e-7;
    double absTol = 1e-7;
    unsigned int order = 8;

    // Set tolerance for tests
    double testTol = 1e-4;
    double lb = 0;
    double ub = 1.0;

    AdaptiveClenshawCurtis quad(maxSub, absTol, relTol,QuadError::First, order);

    TestIntegrand integrand;

    
    // TestIntegrand integrand;
    Kokkos::View<double*, DeviceSpace> dres("integrals", numRepeats);
    Kokkos::parallel_for(numRepeats, KOKKOS_LAMBDA(const unsigned int i){
        dres(i) = quad.Integrate<double>(integrand, lb, ub);
    });

    Kokkos::fence();
    Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);
    double integral = quad.Integrate<double>(integrand, lb, ub);

    for(unsigned int i=0; i<numRepeats; ++i)
        CHECK(hres(i) == Approx(integral).epsilon(1e-3));

}


#endif 