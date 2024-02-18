#include <Kokkos_Core.hpp>
#include <catch2/catch_all.hpp>
#include "MParT/InnerMarginalAffineMap.h"
#include "MParT/IdentityMap.h"
#include "MParT/MapFactory.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

using namespace mpart;
using namespace Catch;
using namespace Catch::Matchers;

using MemorySpace = Kokkos::HostSpace;

TEST_CASE( "Test InnerMarginalAffineMap", "[InnerMarginalAffineMap]") {
    int inputDim = 4;
    double scale_base = 2.3;
    double shift_base = 1.2;
    Kokkos::View<double*, MemorySpace> scale ("Map scale", inputDim);
    Kokkos::View<double*, MemorySpace> shift ("Map shift", inputDim);
    double expected_logdet = 0.;
    for (unsigned int i = 0; i < inputDim; i++) {
        scale(i) = scale_base + i;
        shift(i) = shift_base + i;
        expected_logdet += std::log(scale(i));
    }
    double fd_step = 1e-6;
    SECTION("IdentityMap, square") {
        unsigned int dim = inputDim;
        auto id = std::make_shared<IdentityMap<MemorySpace>>(dim,dim);
        auto map = std::make_shared<InnerMarginalAffineMap<MemorySpace>>(scale, shift, id);
        int N_pts = 10;
        Kokkos::View<double**, MemorySpace> pts("pts", dim, N_pts);
        Kokkos::MDRangePolicy<Kokkos::Rank<2>, typename MemoryToExecution<MemorySpace>::Space> policy({0, 0}, {dim, N_pts});
        Kokkos::parallel_for("Fill pts", policy, KOKKOS_LAMBDA(const unsigned int& i, const unsigned int& j){
            pts(i,j) = i+j;
        });
        // Evaluate
        StridedMatrix<double, MemorySpace> output = map->Evaluate(pts);
        // Check
        for (unsigned int i = 0; i < dim; i++) {
            for (unsigned int j = 0; j < N_pts; j++) {
                REQUIRE_THAT(output(i,j), WithinRel(scale(i)*pts(i,j) + shift(i), 1e-14));
            }
        }
        // LogDeterminant
        StridedVector<double, MemorySpace> logdet = map->LogDeterminant(pts);
        for (unsigned int j = 0; j < N_pts; j++) {
            REQUIRE_THAT(logdet(j), WithinRel(expected_logdet, 1e-14));
        }
        // Inverse
        Kokkos::View<double**, MemorySpace> x1("x1", 0, N_pts);
        StridedMatrix<double, MemorySpace> inv = map->Inverse(x1, pts);
        for (unsigned int i = 0; i < dim; i++) {
            for (unsigned int j = 0; j < N_pts; j++) {
                CHECK_THAT(inv(i,j), WithinRel((pts(i,j) - shift(i))/scale(i), 1e-14));
            }
        }
        // LogDeterminantInputGrad        
        StridedMatrix<double, MemorySpace> logdet_grad = map->LogDeterminantInputGrad(pts);
        for (unsigned int i = 0; i < dim; i++) {
            for (unsigned int j = 0; j < N_pts; j++) {
                REQUIRE_THAT(logdet_grad(i,j), WithinAbs(0., 1e-14));
            }
        }
        // Gradient
        Kokkos::View<double**, MemorySpace> sens("sens", dim, N_pts);
        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<N_pts; ++j){
                sens(i,j) = i+j;
            }
        }
        StridedMatrix<double, MemorySpace> grad = map->Gradient(pts, sens);
        for (unsigned int i = 0; i < dim; i++) {
            for (unsigned int j = 0; j < N_pts; j++) {
                REQUIRE_THAT(grad(i,j), WithinRel(sens(i,j)*scale(i), 1e-14));
            }
        }
    }

    SECTION("TriangularMap, non-square") {
        unsigned int maxOrder = 3;
        unsigned int outputDim = 2;
        unsigned int numPts = 20;
        auto trimap = MapFactory::CreateTriangular<MemorySpace>(inputDim, outputDim, maxOrder, MapOptions());

        // Penalize high order k
        for(unsigned int k = 0; k < trimap->numCoeffs; k++) {
            trimap->Coeffs()(k) = 1/(k+1);
        }
        auto map = std::make_shared<InnerMarginalAffineMap<MemorySpace>>(scale, shift, trimap);

        // Generate points
        Kokkos::View<double**, MemorySpace> pts("pts", inputDim, numPts);
        for(unsigned int i=0; i<inputDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = -1.5+0.22*double(i)+0.1*double(j);
            }
        }

        // Apply the affine part to the points a priori
        Kokkos::View<double**, MemorySpace> appliedPts("appliedPts", inputDim, numPts);
        for(unsigned int i=0; i<inputDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                appliedPts(i,j) = pts(i,j)*scale(i) + shift(i);
            }
        }

        // Evaluate test
        StridedMatrix<double, MemorySpace> exp_output = trimap->Evaluate(appliedPts);
        StridedMatrix<double, MemorySpace> output = map->Evaluate(pts);
        for(unsigned int i=0; i<outputDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                REQUIRE_THAT(output(i,j), WithinRel(exp_output(i,j), 1e-14));
            }
        }

        // LogDeterminant test
        StridedVector<double, MemorySpace> logdet = map->LogDeterminant(pts);
        StridedVector<double, MemorySpace> exp_logdet_map = trimap->LogDeterminant(appliedPts);
        for(unsigned int j=0; j<numPts; ++j){
            // Somewhat circular, but the expression is correct
            CHECK_THAT(logdet(j), WithinRel(exp_logdet_map(j) + expected_logdet, 1e-14));
        }

        // Inverse test
        StridedMatrix<double, MemorySpace> inv = map->Inverse(pts, output);
        for(unsigned int i=0; i<outputDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                REQUIRE_THAT(inv(i,j), WithinRel(pts(i+(inputDim-outputDim), j), 1e-14));
            }
        }

        // LogDeterminantInputGrad test
        StridedMatrix<double, MemorySpace> logdet_grad = map->LogDeterminantInputGrad(pts);
        Kokkos::View<double**, MemorySpace> exp_logdet_grad_map("exp_logdet_grad_map", inputDim, numPts);
        for(int i = 0; i < inputDim; i++) {
            for(int j = 0; j < numPts; j++) {
                appliedPts(i,j) = (pts(i,j)+fd_step)*scale(i) + shift(i);
            }
            StridedVector<double,MemorySpace> exp_logdet_map_fd = trimap->LogDeterminant(appliedPts);
            for(int j = 0; j < numPts; j++) {
                appliedPts(i,j) = (pts(i,j))*scale(i) + shift(i);
            }
            for(int j = 0; j < numPts; j++) {
                exp_logdet_grad_map(i,j) = (exp_logdet_map_fd(j) - exp_logdet_map(j))/fd_step;
            }
        }
        for(unsigned int i=0; i<inputDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                REQUIRE_THAT(logdet_grad(i,j), WithinRel(exp_logdet_grad_map(i,j), 20*fd_step));
            }
        }

        // LogDeterminantCoeffGrad test
        StridedMatrix<double, MemorySpace> logdet_coeff_grad = map->LogDeterminantCoeffGrad(pts);
        Kokkos::View<double**, MemorySpace> exp_logdet_coeff_grad_map("exp_logdet_coeff_grad_map", trimap->numCoeffs, numPts);
        for(int k = 0; k < trimap->numCoeffs; k++) {
            for(int j = 0; j < numPts; j++) {
                trimap->Coeffs()(k) += fd_step;
                StridedVector<double,MemorySpace> exp_logdet_map_fd = trimap->LogDeterminant(appliedPts);
                for(int j = 0; j < numPts; j++) {
                    exp_logdet_coeff_grad_map(k,j) = (exp_logdet_map_fd(j) - exp_logdet_map(j))/fd_step;
                }
                trimap->Coeffs()(k) -= fd_step;
            }
        }

        for(unsigned int k=0; k<trimap->numCoeffs; ++k){
            for(unsigned int j=0; j<numPts; ++j){
                REQUIRE_THAT(logdet_coeff_grad(k,j), WithinRel(exp_logdet_coeff_grad_map(k,j), 20*fd_step));
            }
        }

        Kokkos::View<double**, MemorySpace> sens("sens", outputDim, numPts);
        for(unsigned int i=0; i<outputDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                sens(i,j) = i+j+1;
            }
        }
        // Gradient test (using fd)
        StridedMatrix<double, MemorySpace> grad = map->Gradient(pts, sens);
        Kokkos::View<double**, MemorySpace> exp_grad_map("exp_grad_map", inputDim, numPts);
        for(int i = 0; i < inputDim; i++) {
            for(int j = 0; j < numPts; j++) {
                appliedPts(i,j) = (pts(i,j)+fd_step)*scale(i) + shift(i);
            }
            StridedMatrix<double,MemorySpace> exp_output_fd = trimap->Evaluate(appliedPts);
            for(int j = 0; j < numPts; j++) {
                appliedPts(i,j) = (pts(i,j))*scale(i) + shift(i);
            }
            for(int j = 0; j < numPts; j++) {
                exp_grad_map(i,j) = 0.;
                for(int k = 0; k < outputDim; k++) {
                    exp_grad_map(i,j) += sens(k,j)*(exp_output_fd(k,j) - exp_output(k,j))/fd_step;
                }
            }
        }
        for(unsigned int i=0; i<inputDim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                REQUIRE_THAT(grad(i,j), WithinRel(exp_grad_map(i,j), 20*fd_step) || WithinAbs(exp_grad_map(i,j), 1e-8));
            }
        }

        // CoeffGrad test
        Kokkos::View<double**, MemorySpace> coeff_grad = map->CoeffGrad(pts, sens);
        Kokkos::View<double**, MemorySpace> exp_coeff_grad_map("exp_coeff_grad_map", trimap->numCoeffs, numPts);
        for(int i = 0; i < trimap->numCoeffs; i++) {
            trimap->Coeffs()(i) += fd_step;
            StridedMatrix<double,MemorySpace> exp_output_fd = trimap->Evaluate(appliedPts);
            trimap->Coeffs()(i) -= fd_step;
            for(int j = 0; j < numPts; j++) {
                exp_coeff_grad_map(i,j) = 0.;
                for(int k = 0; k < outputDim; k++) {
                    exp_coeff_grad_map(i,j) += sens(k,j)*(exp_output_fd(k,j) - exp_output(k,j))/fd_step;
                }
            }
        }
        for(unsigned int i=0; i<trimap->numCoeffs; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                REQUIRE_THAT(coeff_grad(i,j), WithinRel(exp_coeff_grad_map(i,j), 20*fd_step) || WithinAbs(exp_coeff_grad_map(i,j), 1e-8));
            }
        }
    }

    SECTION("MoveCoeffs") {
        unsigned int maxOrder = 3;
        unsigned int outputDim = 2;
        unsigned int numPts = 20;
        auto trimap = MapFactory::CreateTriangular<MemorySpace>(inputDim, outputDim, maxOrder, MapOptions());

        // Penalize high order k
        for(unsigned int k = 0; k < trimap->numCoeffs; k++) {
            trimap->Coeffs()(k) = 1/(k+1);
        }
        auto map = std::make_shared<InnerMarginalAffineMap<MemorySpace>>(scale, shift, trimap);
        for(unsigned int k = 0; k < map->numCoeffs; k++) {
            map->Coeffs()(k) += 0.1;
        }
        for(unsigned int k = 0; k < map->numCoeffs; k++) {
            REQUIRE_THAT(map->Coeffs()(k), WithinRel(trimap->Coeffs()(k), 1e-14));
        }
        map = std::make_shared<InnerMarginalAffineMap<MemorySpace>>(scale, shift, trimap, false);
        for(unsigned int k = 0; k < map->numCoeffs; k++) {
            map->Coeffs()(k) += 0.1;
        }
        for(unsigned int k = 0; k < map->numCoeffs; k++) {
            REQUIRE_THAT(map->Coeffs()(k), WithinRel(trimap->Coeffs()(k) + 0.1, 1e-14));
        }
    }
}