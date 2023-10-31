#include <catch2/catch_all.hpp>

#include "MParT/Sigmoid.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/OrthogonalPolynomial.h"

#include "MParT/Utilities/ArrayConversions.h"

#include <Eigen/Dense>

using namespace mpart;
using namespace Catch;

struct TestEvaluators {
    virtual void EvaluateAll(double* output, int max_order, double input) const {
        assert(false);
    };
    virtual void EvaluateDerivatives(double* output, double* output_diff, int max_order, double input) const {
        assert(false);
    };
    virtual void EvaluateSecondDerivatives(double* output, double* output_diff, double* output_diff2, int max_order, double input) const {
        assert(false);
    };
    virtual ~TestEvaluators() = default;
};

struct IdentityEvaluator: public TestEvaluators {
    KOKKOS_INLINE_FUNCTION void EvaluateAll(double* output, int max_order, double input) const override {
        for(int i = 0; i <= max_order; i++) {
            output[i] = input;
        }
    }
    KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(double* output, double* output_diff, int max_order, double input) const override {
        for(int i = 0; i <= max_order; i++) {
            output[i] = input;
            output_diff[i] = 1;
        }
    }
    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(double* output, double* output_diff, double* output_diff2, int max_order, double input) const override {
        for(int i = 0; i <= max_order; i++) {
            output[i] = input;
            output_diff[i] = 1;
            output_diff2[i] = 0;
        }
    }
};

struct NegativeEvaluator: public TestEvaluators {
    KOKKOS_INLINE_FUNCTION void EvaluateAll(double* output, int max_order, double input) const override {
        for(int i = 0; i <= max_order; i++) {
            output[i] = -input;
        }
    }
    KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(double* output, double* output_diff, int max_order, double input) const override {
        for(int i = 0; i <= max_order; i++) {
            output[i] = -input;
            output_diff[i] = -1;
        }
    }
    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(double* output, double* output_diff, double* output_diff2, int max_order, double input) const override {
        for(int i = 0; i <= max_order; i++) {
            output[i] = -input;
            output_diff[i] = -1;
            output_diff2[i] = 0;
        }
    }
};

TEST_CASE( "Testing basis evaluators", "[BasisEvaluators]") {
    int dim = 3;
    const int max_order = 3;
    const double point = 2.3049201;
    double out[max_order+1];
    double out_diff[max_order+1];
    double out_diff2[max_order+1];
    auto reset_out = [&out, &out_diff, &out_diff2](){
        memset(out      , 3., (max_order+1)*sizeof(double));
        memset(out_diff , 3., (max_order+1)*sizeof(double));
        memset(out_diff2, 3., (max_order+1)*sizeof(double));
    };
    SECTION("Homogeneous") {
        reset_out();
        BasisEvaluator<BasisHomogeneity::Homogeneous, IdentityEvaluator> eval {dim,IdentityEvaluator()};
        eval.EvaluateAll(0, out, max_order, point);
        for(int i = 0; i <= max_order; i++) CHECK(out[i] == point);
        
        reset_out();
        eval.EvaluateDerivatives(1, out, out_diff, max_order, point);
        for(int i = 0; i <= max_order; i++) {
            bool is_i_valid = out[i] == point && out_diff[i] == 1.;
            CHECK(is_i_valid);
        }

        reset_out();
        eval.EvaluateSecondDerivatives(2, out, out_diff, out_diff2, max_order, point);
        for(int i = 0; i <= max_order; i++){
            bool is_i_valid = out[i] == point && out_diff[i] == 1. && out_diff2[i] == 0.;
            CHECK(is_i_valid);
        }
    }
    SECTION("OffdiagHomogeneous") {
        using Eval_T = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<NegativeEvaluator, IdentityEvaluator>>;
        Eval_T eval {dim, NegativeEvaluator(), IdentityEvaluator()};
        for(int d = 0; d < dim; d++) {
            reset_out();
            eval.EvaluateAll(d, out, max_order, point);
            for(int i = 0; i <= max_order; i++) CHECK(out[i] == (d == dim-1 ? point : -point));
            
            reset_out();
            eval.EvaluateDerivatives(d, out, out_diff, max_order, point);
            for(int i = 0; i <= max_order; i++) {
                bool is_id_valid = out[i] == (d == dim-1 ? point : -point);
                is_id_valid &= out_diff[i] == (d == dim-1 ? 1 : -1);
                CHECK(is_id_valid);
            }

            reset_out();
            eval.EvaluateSecondDerivatives(d, out, out_diff, out_diff2, max_order, point);
            for(int i = 0; i <= max_order; i++) {
                bool is_id_valid = out[i] == (d == dim-1 ? point : -point);
                is_id_valid &= out_diff[i] == (d == dim-1 ? 1 : -1);
                is_id_valid &= out_diff2[i] == 0.;
                CHECK(is_id_valid);
            }
        }
    }
    SECTION("Heterogeneous") {
        using Vec_T = std::vector<std::shared_ptr<TestEvaluators>>;
        using Eval_T = BasisEvaluator<BasisHomogeneity::Heterogeneous, Vec_T>;
        Vec_T evals1d {std::make_shared<IdentityEvaluator>(), std::make_shared<NegativeEvaluator>(), std::make_shared<IdentityEvaluator>()};
        Eval_T eval {dim, evals1d};
        for(int d = 0; d < dim; d++) {
            reset_out();
            eval.EvaluateAll(d, out, max_order, point);
            for(int i = 0; i <= max_order; i++) CHECK(out[i] == (d % 2 ? -point : point));
            
            reset_out();
            eval.EvaluateDerivatives(d, out, out_diff, max_order, point);
            for(int i = 0; i <= max_order; i++) {
                bool is_id_valid = out[i] == (d % 2 ? -point : point);
                is_id_valid &= out_diff[i] == (d % 2 ? -1: 1);
                CHECK(is_id_valid);
            }

            reset_out();
            eval.EvaluateSecondDerivatives(d, out, out_diff, out_diff2, max_order, point);
            for(int i = 0; i <= max_order; i++) {
                bool is_id_valid = out[i] == (d % 2 ? -point : point);
                is_id_valid &= out_diff[i] == (d % 2 ? -1 : 1);
                is_id_valid &= out_diff2[i] == 0.;
                CHECK(is_id_valid);
            }
        }
    }
}

using HomogeneousEval_T = BasisEvaluator<BasisHomogeneity::Homogeneous, ProbabilistHermite>;
using OffdiagHomogeneousEval_T = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<ProbabilistHermite,Sigmoid1d<SigmoidTypes::Logistic>>>;
using HeterogeneousEval_T = BasisEvaluator<BasisHomogeneity::Heterogeneous, std::vector<std::shared_ptr<ProbabilistHermite>>>;

template<typename T>
T CreateEvaluator(int) {assert(false);}

template<>
HomogeneousEval_T CreateEvaluator<HomogeneousEval_T>(int) {
    return HomogeneousEval_T{};
}

template<>
OffdiagHomogeneousEval_T CreateEvaluator<OffdiagHomogeneousEval_T>(int dim) {
    ProbabilistHermite offdiag;
    const int order = 2;
    double centers[order] = {-1., 2.};
    double widths[order] = {1., 0.5};
    double weights[order] = {0.5, 1.};
    Sigmoid1d diag(order, centers, widths, weights);
    return OffdiagHomogeneousEval_T {dim, offdiag, diag};
}

TEMPLATE_TEST_CASE( "Testing multivariate expansion worker", "[MultivariateExpansionWorker]", HomogeneousEval_T) {

    unsigned int dim = 3;
    unsigned int maxDegree = 3;
    FixedMultiIndexSet<Kokkos::HostSpace> mset(dim, maxDegree); // Create a total order limited fixed multindex set
    TestType poly1d = CreateEvaluator<TestType>(dim);
    MultivariateExpansionWorker<TestType,Kokkos::HostSpace> expansion(mset, poly1d);

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
    double out[maxDegree+1];
    for(unsigned int d=0; d<dim-1;++d){
        for(int i = 0; i <= maxDegree; i++) out[i] = 0.;
        poly1d.EvaluateAll(d, out, maxDegree, pt(d));
        for(unsigned int i=0; i<maxDegree+1; ++i){
            CHECK(cache[i + d*(maxDegree+1)] == Approx(out[i]).epsilon(1e-15) );
        }
    }

    // Fill in the last part of the cache for an evaluation
    expansion.FillCache2(&cache[0], pt, pt(dim-1), DerivativeFlags::None);
    for(int i = 0; i <= maxDegree; i++) out[i] = 0.;
    poly1d.EvaluateAll(dim-1, out, maxDegree, pt(dim-1));
    for(unsigned int i=0; i<maxDegree+1; ++i){
        CHECK(cache[i + (dim-1)*(maxDegree+1)] == Approx(out[i]).epsilon(1e-15) );
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

    SECTION("Retrieve Fixed Mset"){
        FixedMultiIndexSet<Kokkos::HostSpace> mset2 = expansion.GetMultiIndexSet();
        CHECK(mset2.Size() == mset.Size());
        auto mset_max_degrees = mset.MaxDegrees();
        auto mset2_max_degrees = mset2.MaxDegrees();
        CHECK(mset_max_degrees.size() == mset2_max_degrees.size());
        for(unsigned int i=0; i<mset_max_degrees.size(); ++i){
            CHECK(mset_max_degrees[i] == mset2_max_degrees[i]);
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

    MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,Kokkos::HostSpace> hexpansion(hset);
    MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,DeviceSpace> dexpansion(dset);

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