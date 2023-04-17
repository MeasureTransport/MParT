#include <catch2/catch_all.hpp>

#include <Kokkos_Core.hpp>
#include "MParT/AffineMap.h"
#include "MParT/MapFactory.h"
#include "MParT/MapObjective.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

#include <fstream>

using namespace mpart;
using namespace Catch;

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
        CHECK(kl_exact == Approx(kl_est).margin(0.5));
    }

    // Setup map for following sections
    const double coeff_def = 1.;

    auto map = MapFactory::CreateTriangular<Kokkos::HostSpace>(dim, dim, 2);
    Kokkos::parallel_for("Fill coeffs", map->numCoeffs, KOKKOS_LAMBDA(const unsigned int i){
        map->Coeffs()(i) = coeff_def;
    });

    SECTION("CoeffGradImpl"){
        double fd_step = 1e-6;
        double kl_est = objective.ObjectiveImpl(reference_samples, map);
        Kokkos::View<double*, Kokkos::HostSpace> coeffGrad ("Actual CoeffGrad of KL Obj", map->numCoeffs);
        objective.CoeffGradImpl(reference_samples, coeffGrad, map);
        for(int i = 0; i < map->numCoeffs; i++) {
            map->Coeffs()(i) += fd_step;
            double kl_perturb_i = objective.ObjectiveImpl(reference_samples, map);
            double coeffFD_i = (kl_perturb_i - kl_est)/fd_step;
            CHECK(coeffFD_i == Approx(coeffGrad(i)).margin(5*fd_step));
            map->Coeffs()(i) -= fd_step;
        }
    }
    SECTION("ObjectivePlusCoeffGradImpl"){
        double kl_est_ref = objective.ObjectiveImpl(reference_samples, map);
        Kokkos::View<double*, Kokkos::HostSpace> coeffGradRef ("Reference CoeffGrad of KL Obj", map->numCoeffs);
        Kokkos::View<double*, Kokkos::HostSpace> coeffGrad ("CoeffGrad of KL Obj", map->numCoeffs);
        objective.CoeffGradImpl(reference_samples, coeffGradRef, map);
        double kl_est = objective.ObjectivePlusCoeffGradImpl(reference_samples, coeffGrad, map);
        CHECK(kl_est_ref == Approx(kl_est).margin(1e-12));
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK(coeffGradRef(i) == Approx(coeffGrad(i)).margin(1e-12));
        }
    }
    SECTION("MapObjectiveFunctions") {
        // operator()
        Kokkos::View<double*, Kokkos::HostSpace> coeffGradRef ("Reference CoeffGrad of KL Obj", map->numCoeffs);
        Kokkos::View<double*, Kokkos::HostSpace> coeffGrad ("CoeffGrad of KL Obj", map->numCoeffs);
        double kl_est_ref = objective.ObjectivePlusCoeffGradImpl(train_samples, coeffGradRef, map);
        double kl_est = objective(map->numCoeffs, map->Coeffs().data(), coeffGrad.data(), map);
        CHECK(kl_est_ref == Approx(kl_est).margin(1e-12));
        for(int i = 0; i < map->numCoeffs; i++) {
            CHECK(coeffGradRef(i) == Approx(coeffGrad(i)).margin(1e-12));
        }
        // TestError
        double test_error_ref = objective.ObjectiveImpl(test_samples, map);
        double test_error = objective.TestError(map);
        CHECK(test_error_ref == Approx(test_error).margin(1e-12));
        // TrainCoeffGrad
        StridedVector<double,Kokkos::HostSpace> trainCoeffGrad = objective.TrainCoeffGrad(map);
        for(int i = 0; i < map->numCoeffs; i++){
            CHECK(coeffGradRef(i) == Approx(trainCoeffGrad(i)).margin(1e-12));
        }
    }
}

