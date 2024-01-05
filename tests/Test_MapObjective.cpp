#include <catch2/catch_all.hpp>

#include <Kokkos_Core.hpp>
#include "MParT/AffineMap.h"
#include "MParT/MapFactory.h"
#include "MParT/MapObjective.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

#include <fstream>

using namespace mpart;
using namespace Catch;
using namespace Catch::Matchers;
using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Test KLMapObjective", "[KLMapObjective]") {
    unsigned int dim = 2;
    unsigned int seed = 42;
    unsigned int N_samples = 20000;
    unsigned int N_testpts = N_samples/5;
    std::shared_ptr<GaussianSamplerDensity<MemorySpace>> density = std::make_shared<GaussianSamplerDensity<MemorySpace>>(dim);
    density->SetSeed(seed);
    StridedMatrix<double, MemorySpace> reference_samples = density->Sample(N_samples);
    StridedMatrix<double, MemorySpace> test_samples = Kokkos::subview(reference_samples, Kokkos::ALL, Kokkos::make_pair(0u,N_testpts));
    StridedMatrix<double, MemorySpace> train_samples = Kokkos::subview(reference_samples, Kokkos::ALL, Kokkos::make_pair(N_testpts,N_samples));
    KLObjective<MemorySpace> objective {train_samples, test_samples, density};
    ParamL2RegularizationObjective<MemorySpace> param_reg {0.};

    SECTION("ObjectiveImpl") {
        double map_scale = 2.5;
        double map_shift = 1.5;
        Kokkos::View<double**, MemorySpace> A ("Map scale", dim, dim);
        Kokkos::View<double*, MemorySpace> b ("Map shift", dim);
        for(int i = 0; i < dim; i++) {
            b(i) = map_shift;
            for(int j = 0; j < dim; j++) {
                A(i,j) = map_scale * static_cast<double>(i == j);
            }
        }
        auto map = std::make_shared<AffineMap<MemorySpace>>(A,b);
        double kl_est = objective.ObjectiveImpl(reference_samples, map);
        double inv_cov_diag = map_scale*map_scale;
        double kl_exact = -std::log(inv_cov_diag) - 1 + inv_cov_diag + map_shift*map_shift*inv_cov_diag;
        kl_exact /= 2.;
        CHECK_THAT(kl_est, WithinRel(kl_exact, 0.5));
    }

    // Setup map for following sections
    auto map = MapFactory::CreateTriangular<MemorySpace>(dim, dim, 2);
    double scale = 1.3423;
    double coeff_sq_norm = 0.;
    for(int i = 0; i < map->numCoeffs; i++) {
        double coeff = (i+1)*scale;
        map->Coeffs()(i) = coeff;
        coeff_sq_norm += coeff*coeff;
    }

    SECTION("CoeffGradImpl"){
        double fd_step = 1e-8;
        double kl_est = objective.ObjectiveImpl(reference_samples, map);
        Kokkos::View<double*, MemorySpace> coeffGrad ("Actual CoeffGrad of KL Obj", map->numCoeffs);
        objective.CoeffGradImpl(reference_samples, coeffGrad, map);
        for(int i = 0; i < map->numCoeffs; i++) {
            double prev = map->Coeffs()(i);
            map->Coeffs()(i) += fd_step;
            double kl_perturb_i = objective.ObjectiveImpl(reference_samples, map);
            double coeffFD_i = (kl_perturb_i - kl_est)/fd_step;
            CHECK_THAT(coeffFD_i, WithinRel(coeffGrad(i), 1e-3));
            map->Coeffs()(i) = prev;
        }
    }
    SECTION("ObjectivePlusCoeffGradImpl"){
        double kl_est_ref = objective.ObjectiveImpl(reference_samples, map);
        Kokkos::View<double*, MemorySpace> coeffGradRef ("Reference CoeffGrad of KL Obj", map->numCoeffs);
        Kokkos::View<double*, MemorySpace> coeffGrad ("CoeffGrad of KL Obj", map->numCoeffs);
        objective.CoeffGradImpl(reference_samples, coeffGradRef, map);
        double kl_est = objective.ObjectivePlusCoeffGradImpl(reference_samples, coeffGrad, map);
        CHECK_THAT(kl_est_ref, WithinRel(kl_est, 1e-12));
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK_THAT(coeffGradRef(i), WithinRel(coeffGrad(i), 1e-12));
        }
    }

    SECTION("ParamRegularization"){
        const int n_param = map->numCoeffs;
        Kokkos::View<double*, MemorySpace> param_grad(  "ParamReg Coeff Grad", n_param);
        Kokkos::View<double*, MemorySpace> param_grad_2("ParamReg Coeff Grad", n_param);
        param_reg.SetScale(0.);
        double param_reg_eval_0 = param_reg.ObjectiveImpl(reference_samples, map);
        param_reg.CoeffGradImpl(reference_samples, param_grad, map);
        double param_reg_eval_0_2 = param_reg.ObjectivePlusCoeffGradImpl(reference_samples, param_grad_2, map);
        CHECK(param_reg_eval_0   == 0.);
        CHECK(param_reg_eval_0_2 == 0.);
        for(int i = 0; i < n_param; i++) CHECK(param_grad(i)   == 0.);
        for(int i = 0; i < n_param; i++) CHECK(param_grad_2(i) == 0.);
        param_reg.SetScale(1.);
        double param_reg_eval_1 = param_reg.ObjectiveImpl(reference_samples, map);
        param_reg.CoeffGradImpl(reference_samples, param_grad, map);
        double param_reg_eval_1_2 = param_reg.ObjectivePlusCoeffGradImpl(reference_samples, param_grad_2, map);
        CHECK_THAT(param_reg_eval_1,   WithinRel(coeff_sq_norm, 1e-12));
        CHECK_THAT(param_reg_eval_1_2, WithinRel(coeff_sq_norm, 1e-12));
        for(int i = 0; i < n_param; i++) CHECK_THAT(param_grad(i),   WithinRel(2*map->Coeffs()(i)));
        for(int i = 0; i < n_param; i++) CHECK_THAT(param_grad_2(i), WithinRel(2*map->Coeffs()(i)));
        double arbitrary_scale = 2e-4;
        param_reg.SetScale(arbitrary_scale);
        double param_reg_eval_arb = param_reg.ObjectiveImpl(reference_samples, map);
        param_reg.CoeffGradImpl(reference_samples, param_grad, map);
        double param_reg_eval_arb_2 = param_reg.ObjectivePlusCoeffGradImpl(reference_samples, param_grad_2, map);
        CHECK_THAT(param_reg_eval_arb,   WithinRel(coeff_sq_norm*arbitrary_scale, 1e-12));
        CHECK_THAT(param_reg_eval_arb_2, WithinRel(coeff_sq_norm*arbitrary_scale, 1e-12));
        for(int i = 0; i < n_param; i++) CHECK_THAT(param_grad(i),   WithinRel(2*arbitrary_scale*map->Coeffs()(i)));
        for(int i = 0; i < n_param; i++) CHECK_THAT(param_grad_2(i), WithinRel(2*arbitrary_scale*map->Coeffs()(i)));
    }

    SECTION("MapObjectiveFunctions") {
        // operator()
        Kokkos::View<double*, MemorySpace> coeffGradRef ("Reference CoeffGrad of KL Obj", map->numCoeffs);
        Kokkos::View<double*, MemorySpace> coeffGrad ("CoeffGrad of KL Obj", map->numCoeffs);
        double kl_est_ref = objective.ObjectivePlusCoeffGradImpl(train_samples, coeffGradRef, map);
        double kl_est = objective(map->numCoeffs, map->Coeffs().data(), coeffGrad.data(), map);
        CHECK_THAT(kl_est, WithinRel(kl_est_ref, 1e-12));
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK_THAT(coeffGradRef(i), WithinAbs(coeffGrad(i), 1e-12));
        }
        // TestError
        double test_error_ref = objective.ObjectiveImpl(test_samples, map);
        double test_error = objective.TestError(map);
        CHECK_THAT(test_error, WithinRel(test_error_ref, 1e-12));
        // TrainCoeffGrad
        StridedVector<double,MemorySpace> trainCoeffGrad = objective.TrainCoeffGrad(map);
        for(int i = 0; i < map->numCoeffs; i++){
            CHECK_THAT(coeffGradRef(i), WithinRel(trainCoeffGrad(i), 1e-12));
        }
    }
}

