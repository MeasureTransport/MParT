#include <catch2/catch_all.hpp>

#include "MParT/Quadrature.h"
#include "MParT/Utilities/ArrayConversions.h"

#include "MParT/Utilities/KokkosSpaceMappings.h"

using namespace mpart;
using namespace Catch;


class TestIntegrand {
public:

    KOKKOS_FUNCTION TestIntegrand(double shift=0) : shift_(shift){};

    KOKKOS_INLINE_FUNCTION void operator()(double x, double* f) const{
        f[0] = exp(x) + shift_;
    }

    const double shift_;
}; // class TestIntegrand

TEST_CASE("Testing Nested CC Quadrature", "[NestedClenshawCurtis]") {

    unsigned int nestedLevel = 3;
    unsigned int coarseNum = std::pow(2,nestedLevel) + 1;
    unsigned int fineNum = std::pow(2,nestedLevel+1) + 1;
    
    Kokkos::View<double*,Kokkos::HostSpace> wts1("Coarse Wts", coarseNum);
    Kokkos::View<double*,Kokkos::HostSpace> pts1("Coarse Pts", coarseNum);
    ClenshawCurtisQuadrature<Kokkos::HostSpace>::GetRule(coarseNum, wts1.data(), pts1.data());

    Kokkos::View<double*,Kokkos::HostSpace> wts2("Coarse Wts", fineNum);
    Kokkos::View<double*,Kokkos::HostSpace> pts2("Coarse Pts", fineNum);
    ClenshawCurtisQuadrature<Kokkos::HostSpace>::GetRule(fineNum, wts2.data(), pts2.data());


    for(unsigned int i=0; i<coarseNum; ++i){
        CHECK(pts1(i) == pts2(2*i));
    }
}


TEST_CASE( "Testing CC Quadrature", "[ClenshawCurtisQuadrature]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int order = 10;

    // Set tolerance for tests
    double testTol = 1e-8;

    ClenshawCurtisQuadrature quad(order,2);

    SECTION("Class Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        TestIntegrand integrand;
        double integral;
        quad.SetDim(1);
        quad.Integrate(integrand, lb, ub, &integral);

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

    SECTION("Lambda Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        auto integrand = [](double x, double* f){f[0]=exp(x);};
        double integral;
        quad.SetDim(1);
        quad.Integrate(integrand, lb, ub, &integral);    

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

    SECTION("Vector-Valued Integrand")
    {
        double lb = 0.0;
        double ub = 1.0;

        auto integrand = [](double x, double* f){f[0] = exp(x); f[1]=2.0*exp(x);};

        double integral[2];
        quad.SetDim(2);
        quad.Integrate(integrand, lb, ub, integral);    

        CHECK( integral[0] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( integral[1] == Approx(2.0*(exp(ub)-exp(lb))).epsilon(testTol) );
    }

    SECTION("Vector-Valued with External Workspace")
    {   
        ClenshawCurtisQuadrature quad2(order,2);
        
        std::vector<double> workspace(quad2.WorkspaceSize());
        quad2.SetWorkspace(&workspace[0]);
        
        double lb = 0.0;
        double ub = 1.0;

        auto integrand = [](double x, double* f){f[0]=exp(x); f[1]=exp(x);};

        double integral[2];
        quad2.Integrate(integrand, lb, ub, integral);    

        CHECK( integral[0] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( integral[1] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

}


TEST_CASE( "Testing Adaptive Clenshaw-Curtis Quadrature", "[AdaptiveClenshawCurtis]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int maxSub = 10;
    unsigned int maxDim = 2;

    double relTol = 1e-6;
    double absTol = 1e-6;

    // Set tolerance for tests
    double testTol = 1e-4;

    AdaptiveClenshawCurtis quad(2, maxSub, maxDim, absTol, relTol, QuadError::First);
    quad.SetDim(1); // The maximum dimension of f is 2, but the first few examples only have a dimension of 1

    SECTION("Class Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        TestIntegrand integrand;
        double integral;
        quad.Integrate(integrand, lb, ub, &integral);

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

    SECTION("Lambda Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        auto integrand = [](double x, double* f){f[0]=exp(x);};
        double integral;
        quad.Integrate(integrand, lb, ub, &integral);    

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }


    SECTION("Discontinuous Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        unsigned int numEvals = 0;

        auto integrand = [&](double x, double* f){
            numEvals++;
            if(x<0.5)
                f[0]=exp(x);
            else 
                f[0]=1.0+exp(x);
        };
        double integral;
        quad.Integrate(integrand, lb, ub, &integral);    

        double trueVal = (ub-0.5) + exp(ub)-exp(lb);
        CHECK( integral == Approx(trueVal).epsilon(testTol) );
        CHECK( numEvals<300);
    }

    SECTION("Vector-Valued Integrand")
    {
        double lb = 0.0;
        double ub = 1.0;
        quad.SetDim(2);

        auto integrand = [](double x, double* f){f[0]=exp(x); f[1]=exp(x);};

        double integral[2];
        quad.Integrate(integrand, lb, ub, integral);    

        CHECK( integral[0] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( integral[1] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

    SECTION("Vector-Valued with External Workspace")
    {       
        unsigned int level = 2;
        Kokkos::View<double*, Kokkos::HostSpace> cPts("Coarse Points", std::pow(2,level)+1);
        Kokkos::View<double*, Kokkos::HostSpace> cWts("Coarse Weights", std::pow(2,level)+1);
        Kokkos::View<double*, Kokkos::HostSpace> fPts("Fine Points", std::pow(2,level+1)+1);
        Kokkos::View<double*, Kokkos::HostSpace> fWts("Fine Weights", std::pow(2,level+1)+1);

        AdaptiveClenshawCurtis quad2(cPts,cWts,fPts,fWts, maxSub, maxDim, nullptr, absTol, relTol, QuadError::First);
        
        std::vector<double> workspace(quad2.WorkspaceSize());
        quad2.SetWorkspace(&workspace[0]);

        double lb = 0.0;
        double ub = 1.0;

        auto integrand = [](double x, double* f){f[0]=exp(x); f[1]=exp(x);};

        double integral[2];
        quad2.Integrate(integrand, lb, ub, integral);    

        CHECK( integral[0] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( integral[1] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }
}



TEST_CASE( "Testing Adaptive Simpson Integration", "[AdaptiveSimpson]" ) {

    // Set parameters for adaptive quadrature algorithm
    unsigned int maxSub = 30;
    unsigned int maxDim = 2;

    double relTol = 1e-8;
    double absTol = 1e-8;

    // Set tolerance for tests
    double testTol = 1e-4;

    AdaptiveSimpson quad(maxSub, maxDim, absTol, relTol, QuadError::First);
    quad.SetDim(1); // The maximum dimension of f is 2, but the first few examples only have a dimension of 1

    SECTION("Class Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        TestIntegrand integrand;
        double integral;
        quad.Integrate(integrand, lb, ub, &integral);

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

    SECTION("Lambda Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        auto integrand = [](double x, double* f){f[0]=exp(x);};
        double integral;
        quad.Integrate(integrand, lb, ub, &integral);    

        CHECK( integral == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }


    SECTION("Discontinuous Integrand")
    {   
        double lb = 0;
        double ub = 1.0;

        unsigned int numEvals = 0;

        auto integrand = [&](double x, double* f){
            numEvals++;
            if(x<0.5)
                f[0]= exp(x);
            else 
                f[0]=1.0+exp(x);
        };
        double integral;
        quad.Integrate(integrand, lb, ub, &integral);    

        double trueVal = (ub-0.5) + exp(ub)-exp(lb);
        CHECK( integral == Approx(trueVal).epsilon(testTol) );
        CHECK( numEvals<150);
    }

    SECTION("Vector-Valued Integrand")
    {
        double lb = 0.0;
        double ub = 1.0;
        quad.SetDim(2);

        auto integrand = [](double x, double* f){f[0]=exp(x); f[1]=exp(x);};

        double integral[2];
        quad.Integrate(integrand, lb, ub, integral);    

        CHECK( integral[0] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( integral[1] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }

    SECTION("Vector-Valued with External Workspace")
    {   
        AdaptiveSimpson quad2(maxSub, maxDim, nullptr, absTol, relTol, QuadError::First);
        
        std::vector<double> workspace(quad2.WorkspaceSize());
        quad2.SetWorkspace(&workspace[0]);

        double lb = 0.0;
        double ub = 1.0;

        auto integrand = [](double x, double* f){f[0]=exp(x); f[1]=exp(x);};

        double integral[2];
        quad2.Integrate(integrand, lb, ub, integral);    

        CHECK( integral[0] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
        CHECK( integral[1] == Approx(exp(ub)-exp(lb)).epsilon(testTol) );
    }
}




#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)


TEST_CASE( "Testing CC Quadrature on device", "[ClenshawCurtisDevice]" ) {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;
    typedef typename MemoryToExecution<DeviceSpace>::Space ExecutionSpace;

    // Set parameters for adaptive quadrature algorithm
    unsigned int order = 5;
    unsigned int numRepeats = 320;

    // Set tolerance for tests
    double lb = 0;
    double ub = 1.0;

    
    ClenshawCurtisQuadrature<DeviceSpace> quad(order, 1, nullptr);
    
    // TestIntegrand integrand;
    Kokkos::View<double*, DeviceSpace> dres("integrals", numRepeats);
    unsigned int workspaceSize = quad.WorkspaceSize();

    auto cacheBytes = Kokkos::View<double*,DeviceSpace>::shmem_size(workspaceSize);
    auto policy = Kokkos::TeamPolicy<ExecutionSpace>(numRepeats/32, 32).set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));

    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member){
        unsigned int i = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

        Kokkos::View<double*,DeviceSpace> workspace(team_member.thread_scratch(1), workspaceSize);
        TestIntegrand integrand(i);
        quad.Integrate(workspace.data(), integrand, lb, ub, &dres(i));
    });

    Kokkos::fence();
    Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);

    ClenshawCurtisQuadrature<Kokkos::HostSpace> quad2(order, 1);
    double integral;
    TestIntegrand integrand(0);
    quad2.Integrate(integrand, lb, ub, &integral);

    for(unsigned int i=0; i<numRepeats; ++i)
        CHECK(hres(i) == Approx(integral+i*(ub-lb)).epsilon(1e-7));

}



TEST_CASE( "Testing Adaptive Simpson Quadrature on device", "[AdaptiveSimpsonDevice]" ) {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;
    typedef typename MemoryToExecution<DeviceSpace>::Space ExecutionSpace;

    unsigned int numRepeats = 64;

    // Set parameters for adaptive quadrature algorithm
    unsigned int maxSub = 30;
    double relTol = 1e-6;
    double absTol = 1e-6;
    
    // Set tolerance for tests
    double lb = 0;
    double ub = 1.0;

    AdaptiveSimpson<DeviceSpace> quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

    // TestIntegrand integrand;
    Kokkos::View<double*, DeviceSpace> dres("integrals", numRepeats);
    unsigned int workspaceSize = quad.WorkspaceSize();

    auto cacheBytes = Kokkos::View<double*,DeviceSpace>::shmem_size(workspaceSize);
    auto policy = Kokkos::TeamPolicy<ExecutionSpace>(numRepeats/32, 32).set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));

    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member){
        unsigned int i = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

        Kokkos::View<double*,DeviceSpace> workspace(team_member.thread_scratch(1), workspaceSize);

        TestIntegrand integrand(i);
        quad.Integrate(workspace.data(), integrand, lb, ub, &dres(i));
    });

    Kokkos::fence();
    Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);
    
    AdaptiveSimpson<Kokkos::HostSpace> quad2(maxSub, 1, absTol, relTol, QuadError::First);
    double integral;
    TestIntegrand integrand;
    quad2.Integrate(integrand, lb, ub, &integral);

    for(unsigned int i=0; i<numRepeats; ++i)
        CHECK(hres(i) == Approx(integral + i*(ub-lb)).epsilon(1e-3));

}


TEST_CASE( "Testing Adaptive Clenshaw Curtis on device", "[AdaptiveCCDevice]" ) {

    typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;
    typedef typename MemoryToExecution<DeviceSpace>::Space ExecutionSpace;

    unsigned int numRepeats = 320;

    unsigned int maxSub = 10;
    double relTol = 1e-7;
    double absTol = 1e-7;
    unsigned int order = 8;

    // Set tolerance for tests
    double lb = 0;
    double ub = 1.0;

    AdaptiveClenshawCurtis<DeviceSpace> quad(order, maxSub, 1, nullptr, absTol, relTol, QuadError::First);

    Kokkos::View<double*, DeviceSpace> dres("integrals", numRepeats);
    unsigned int workspaceSize = quad.WorkspaceSize();

    auto cacheBytes = Kokkos::View<double*,DeviceSpace>::shmem_size(workspaceSize);
    auto policy = Kokkos::TeamPolicy<ExecutionSpace>(numRepeats/32, 32).set_scratch_size(1,Kokkos::PerTeam(0), Kokkos::PerThread(cacheBytes));

    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(typename Kokkos::TeamPolicy<ExecutionSpace>::member_type team_member){
        unsigned int i = team_member.league_rank () * team_member.team_size () + team_member.team_rank ();

        Kokkos::View<double*,DeviceSpace> workspace(team_member.thread_scratch(1), workspaceSize);

        TestIntegrand integrand(i);
        quad.Integrate(workspace.data(), integrand, lb, ub, &dres(i));
    });

    Kokkos::fence();
    Kokkos::View<double*, Kokkos::HostSpace> hres = ToHost(dres);

    AdaptiveClenshawCurtis<Kokkos::HostSpace> quad2(order, maxSub, 1, absTol, relTol,QuadError::First); 
    TestIntegrand integrand;   
    double integral;
    quad2.Integrate(integrand, lb, ub, &integral);

    for(unsigned int i=0; i<numRepeats; ++i)
        CHECK(hres(i) == Approx(integral+i*(ub-lb)).epsilon(1e-3));
}


#endif 