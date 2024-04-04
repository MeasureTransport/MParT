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

TEST_CASE( "Test KLMapObjective", "[KLMapObjective]") {
    unsigned int dim = 2;
    unsigned int seed = 42;
    unsigned int N_samples = 20000;
    unsigned int N_testpts = N_samples/5;
    std::shared_ptr<GaussianSamplerDensity<Kokkos::HostSpace>> density = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(dim);
    density->SetSeed(seed);
    StridedMatrix<double, Kokkos::HostSpace> reference_samples = density->Sample(N_samples);
    StridedMatrix<double, Kokkos::HostSpace> test_samples = Kokkos::subview(reference_samples, Kokkos::ALL, Kokkos::make_pair(0u,N_testpts));
    StridedMatrix<double, Kokkos::HostSpace> train_samples = Kokkos::subview(reference_samples, Kokkos::ALL, Kokkos::make_pair(N_testpts,N_samples));
    KLObjective<Kokkos::HostSpace> objective {train_samples, test_samples, density};

    SECTION("ObjectiveImpl") {
        double map_scale = 2.5;
        double map_shift = 1.5;
        Kokkos::View<double**, Kokkos::HostSpace> A ("Map scale", dim, dim);
        Kokkos::View<double*, Kokkos::HostSpace> b ("Map shift", dim);
        for(int i = 0; i < dim; i++) {
            b(i) = map_shift;
            for(int j = 0; j < dim; j++) {
                A(i,j) = map_scale * static_cast<double>(i == j);
            }
        }
        auto map = std::make_shared<AffineMap<Kokkos::HostSpace>>(A,b);
        double kl_est = objective.ObjectiveImpl(reference_samples, map);
        double inv_cov_diag = map_scale*map_scale;
        double kl_exact = -std::log(inv_cov_diag) - 1 + inv_cov_diag + map_shift*map_shift*inv_cov_diag;
        kl_exact /= 2.;
        CHECK_THAT(kl_exact, WithinRel(kl_est, 0.5));
    }

    // Setup map for following sections
    const double coeff_def = 1.;

    auto map = MapFactory::CreateTriangular<Kokkos::HostSpace>(dim, dim, 2);
    Kokkos::deep_copy(map->Coeffs(), coeff_def);

    SECTION("CoeffGradImpl"){
        double fd_step = 1e-6;
        double kl_est = objective.ObjectiveImpl(reference_samples, map);
        Kokkos::View<double*, Kokkos::HostSpace> coeffGrad ("Actual CoeffGrad of KL Obj", map->numCoeffs);
        objective.CoeffGradImpl(reference_samples, coeffGrad, map);
        for(int i = 0; i < map->numCoeffs; i++) {
            map->Coeffs()(i) += fd_step;
            double kl_perturb_i = objective.ObjectiveImpl(reference_samples, map);
            double coeffFD_i = (kl_perturb_i - kl_est)/fd_step;
            CHECK_THAT(coeffFD_i, WithinRel(coeffGrad(i), 10*fd_step));
            map->Coeffs()(i) -= fd_step;
        }
    }
    SECTION("ObjectivePlusCoeffGradImpl"){
        double kl_est_ref = objective.ObjectiveImpl(reference_samples, map);
        Kokkos::View<double*, Kokkos::HostSpace> coeffGradRef ("Reference CoeffGrad of KL Obj", map->numCoeffs);
        Kokkos::View<double*, Kokkos::HostSpace> coeffGrad ("CoeffGrad of KL Obj", map->numCoeffs);
        objective.CoeffGradImpl(reference_samples, coeffGradRef, map);
        double kl_est = objective.ObjectivePlusCoeffGradImpl(reference_samples, coeffGrad, map);
        CHECK_THAT(kl_est_ref, WithinRel(kl_est, 1e-12));
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK_THAT(coeffGradRef(i), WithinRel(coeffGrad(i), 1e-12));
        }
    }
    SECTION("MapObjectiveFunctions") {
        // operator()
        Kokkos::View<double*, Kokkos::HostSpace> coeffGradRef ("Reference CoeffGrad of KL Obj", map->numCoeffs);
        Kokkos::View<double*, Kokkos::HostSpace> coeffGrad ("CoeffGrad of KL Obj", map->numCoeffs);
        double kl_est_ref = objective.ObjectivePlusCoeffGradImpl(train_samples, coeffGradRef, map);
        double kl_est = objective(map->numCoeffs, map->Coeffs().data(), coeffGrad.data(), map);
        CHECK_THAT(kl_est_ref, WithinRel(kl_est, 1e-12));
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK_THAT(coeffGradRef(i), WithinRel(coeffGrad(i), 1e-12));
        }
        // TestError
        double test_error_ref = objective.ObjectiveImpl(test_samples, map);
        double test_error = objective.TestError(map);
        CHECK_THAT(test_error_ref, WithinRel(test_error, 1e-12));
        // TrainCoeffGrad
        StridedVector<double,Kokkos::HostSpace> trainCoeffGrad = objective.TrainCoeffGrad(map);
        for(int i = 0; i < map->numCoeffs; i++){
            CHECK_THAT(coeffGradRef(i), WithinRel(trainCoeffGrad(i), 1e-12));
        }
    }
}

TEST_CASE( "Test FastGaussianReverseKLObjective", "[FastGaussianReverseKLObjective]") {
    unsigned int dim = 1;
    unsigned int maxOrder = 4;
    auto map = MapFactory::CreateTriangular<Kokkos::HostSpace>(dim, dim, maxOrder);

    unsigned int seed = 42;
    unsigned int N_samples = 20000;
    unsigned int N_testpts = N_samples/5;

    std::shared_ptr<GaussianSamplerDensity<Kokkos::HostSpace>> density = std::make_shared<GaussianSamplerDensity<Kokkos::HostSpace>>(dim);
    density->SetSeed(seed);
    StridedMatrix<double, Kokkos::HostSpace> reference_samples = density->Sample(N_samples);
    StridedMatrix<double, Kokkos::HostSpace> test_samples = Kokkos::subview(reference_samples, Kokkos::ALL, Kokkos::make_pair(0u,N_testpts));
    StridedMatrix<double, Kokkos::HostSpace> train_samples = Kokkos::subview(reference_samples, Kokkos::ALL, Kokkos::make_pair(N_testpts,N_samples));

    FastGaussianReverseKLObjective<Kokkos::HostSpace> objective {train_samples, test_samples, map->numCoeffs};
    KLObjective<Kokkos::HostSpace> klObjective {train_samples, test_samples, density};
    Kokkos::View<double*, Kokkos::HostSpace> coeffGradRef ("Reference CoeffGrad of KL Obj", map->numCoeffs);
    Kokkos::View<double*, Kokkos::HostSpace> coeffGrad ("CoeffGrad of KL Obj", map->numCoeffs);
    Kokkos::deep_copy(map->Coeffs(), 0.1);

    SECTION("ObjectivePlusCoeffGradImpl"){
        double kl_err_ref = klObjective.ObjectivePlusCoeffGradImpl(train_samples, coeffGradRef, map);
        double kl_err = objective.ObjectivePlusCoeffGradImpl(train_samples, coeffGrad, map);
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK_THAT(coeffGradRef(i), WithinRel(coeffGrad(i), 1e-12));
        }
        Kokkos::deep_copy(map->Coeffs(), 0.5);
        double kl_err_ref2 = klObjective.ObjectivePlusCoeffGradImpl(train_samples, coeffGradRef, map);
        double kl_err2 = objective.ObjectivePlusCoeffGradImpl(train_samples, coeffGrad, map);
        double slope = (kl_err_ref2 - kl_err_ref) / (kl_err2 - kl_err);
        CHECK_THAT(slope, WithinRel(1., 1e-12));
        double shift = kl_err_ref;
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK_THAT(coeffGradRef(i), WithinRel(coeffGrad(i), 1e-12));
        }
        Kokkos::deep_copy(map->Coeffs(), -0.1);
        double kl_err_ref3 = klObjective.ObjectivePlusCoeffGradImpl(train_samples, coeffGradRef, map);
        double kl_err3 = objective.ObjectivePlusCoeffGradImpl(train_samples, coeffGrad, map);
        double pred_ref = (kl_err3-kl_err)*slope + shift;
        CHECK_THAT(kl_err_ref3, WithinRel(pred_ref, 1e-12));
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK_THAT(coeffGradRef(i), WithinRel(coeffGrad(i), 1e-12));
        }
    }
    SECTION("ObjectiveImpl") {
        double kl_err_cgrad = objective.ObjectivePlusCoeffGradImpl(train_samples, coeffGrad, map);
        double kl_err = objective.ObjectiveImpl(train_samples, map);
        CHECK_THAT(kl_err_cgrad, WithinRel(kl_err, 1e-14));
    }
    SECTION("CoeffGradImpl") {
        Kokkos::deep_copy(coeffGrad, 0.);
        Kokkos::deep_copy(coeffGradRef, 0.);
        objective.ObjectivePlusCoeffGradImpl(train_samples, coeffGradRef, map);
        objective.CoeffGradImpl(train_samples, coeffGrad, map);
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK_THAT(coeffGradRef(i), WithinRel(coeffGrad(i), 1e-14));
        }
    }
}