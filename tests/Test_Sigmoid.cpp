#include <catch2/catch_all.hpp>
#include "MParT/Sigmoid.h"

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

template<typename Function>
void TestSigmoidGradients(Function Sigmoid, unsigned int N_grad_points, double fd_delta, Kokkos::View<double*, MemorySpace> coeff) {
        Kokkos::View<double**, MemorySpace> gradPts("Gradient points", 1, N_grad_points);
        Kokkos::View<double**, MemorySpace> gradPts_plus_delta("Gradient points plus delta", 1, N_grad_points);
        Kokkos::View<double**, MemorySpace> sens("Sensitivities", 1, N_grad_points);
        Kokkos::parallel_for(N_grad_points, KOKKOS_LAMBDA(unsigned int point_index) {
            double gradPt = 3.0*(-1.0 + 2*((double) point_index)/((double) N_grad_points-1));
            double sensPt = 2.0 + ((double) point_index)/((double) N_grad_points-1);
            gradPts(0,point_index) = gradPt;
            gradPts_plus_delta(0,point_index) = gradPt + fd_delta;
            sens(0,point_index) = sensPt;
        });
        Kokkos::View<double**, MemorySpace> input_grad = Sigmoid.Gradient(sens, gradPts);
        Kokkos::View<double**, MemorySpace> coeff_grad = Sigmoid.CoeffGrad(sens, gradPts);
        Kokkos::View<double**, MemorySpace> gradPts_eval = Sigmoid.Evaluate(gradPts);
        Kokkos::View<double**, MemorySpace> gradPts_plus_delta_eval = Sigmoid.Evaluate(gradPts_plus_delta);
        Kokkos::View<double**, MemorySpace> gradPts_coeff_plus_delta_eval ("Evals for each directional add", coeff.extent(0), N_grad_points);
        for(int j = 0; j < coeff.extent(0); j++) {
            coeff(j) += fd_delta;
            Kokkos::View<double**, MemorySpace> coeff_eval_perturb_j = Sigmoid.Evaluate(gradPts);
            Kokkos::parallel_for(N_grad_points, KOKKOS_LAMBDA(unsigned int i){
                gradPts_coeff_plus_delta_eval(j,i) = coeff_eval_perturb_j(0,i);
            });
            coeff(j) -= fd_delta;
        }
        double input_grad_error_accumulator = 0., coeff_grad_error_accumulator = 0.;
        for(int i = 0; i < N_grad_points; i++) {
            double input_grad_i_fd = (gradPts_plus_delta_eval(0,i) - gradPts_eval(0,i))/fd_delta;
            input_grad_error_accumulator += fabs(input_grad_i_fd*sens(0,i) - input_grad(0,i));
            for(int j = 0; j < coeff.extent(0); j++) {
                double coeff_grad_ij_fd = (gradPts_coeff_plus_delta_eval(j,i) - gradPts_eval(0,i))/fd_delta;
                coeff_grad_error_accumulator += fabs(coeff_grad_ij_fd*sens(0,i) - coeff_grad(j,i));
            }
        }
        CHECK(input_grad_error_accumulator < 3*N_grad_points*fd_delta);
        CHECK(coeff_grad_error_accumulator < 3*N_grad_points*coeff.extent(0)*fd_delta);
}

TEMPLATE_TEST_CASE("Sigmoid1d","[sigmoid1d]", Logistic) {
    SECTION("Initialization") {
        Kokkos::View<double*, MemorySpace> centers("Sigmoid Centers", 2);
        Kokkos::View<double*, MemorySpace> widths("Sigmoid Centers", 1);

        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths)), std::invalid_argument);
    }

    unsigned int N_grad_points = 100;
    double fd_delta = 1e-6;
    const double support_bound = 100.;

    SECTION("Single Sigmoid") {
        Kokkos::View<double*, MemorySpace> center("Sigmoid Centers", 1);
        Kokkos::View<double*, MemorySpace> width("Sigmoid Centers", 1);
        Kokkos::View<double*, MemorySpace> coeff("Sigmoid coeff", 1);
        center(0) = 0; width(0) = 1;
        Sigmoid1d<MemorySpace,TestType> Sigmoid (center, width);
        for(int coeff_int = 1; coeff_int <= 2; coeff_int++) {
            coeff(0) = (double) coeff_int;
            Sigmoid.WrapCoeffs(coeff);
            Kokkos::View<double**, MemorySpace> evalPts("Input point", 1, 3);
            evalPts(0,0) = -support_bound; evalPts(0,1) = 0.0; evalPts(0,2) = support_bound;
            StridedMatrix<double,MemorySpace> out = Sigmoid.Evaluate(evalPts);
            double approxTol = 1e-5;
            REQUIRE_THAT(out(0,0), Matchers::WithinAbs(coeff_int*0.0, approxTol));
            REQUIRE_THAT(out(0,1), Matchers::WithinAbs(coeff_int*0.5, approxTol));
            REQUIRE_THAT(out(0,2), Matchers::WithinAbs(coeff_int*1.0, approxTol));
        }
        TestSigmoidGradients(Sigmoid, N_grad_points, fd_delta, coeff);
    }
    SECTION("Multiple Sigmoids") {
        int N_sigmoid = 3;
        Kokkos::View<double*, MemorySpace> centers("Sigmoid Centers", N_sigmoid);
        Kokkos::View<double*, MemorySpace> widths("Sigmoid Centers", N_sigmoid);
        Sigmoid1d<MemorySpace,TestType> Sigmoid (centers, widths);
        Kokkos::View<double*, MemorySpace> coeffs("Sigmoid Coefficients", N_sigmoid);
        for(int j = 0; j < N_sigmoid; j++) {
            centers(j) = 1-0.75*j;
            widths(j) = 2*(j+1);
        }
        Sigmoid.WrapCoeffs(coeffs);
        for(int coeff_int = 1; coeff_int <= 2; coeff_int++) {
            double coeff_sum = 0.;
            for(int j = 1; j < N_sigmoid; j++) {
                double coeff_j = 0.5*((double) N_sigmoid*(j+1));
                coeffs(j) = coeff_j;
                coeff_sum += coeff_j;
            }
            Kokkos::View<double**, MemorySpace> evalPts("Input point", 1, N_sigmoid+2);
            evalPts(0,0) = -support_bound; evalPts(0,1) = support_bound;
            Kokkos::View<double*, MemorySpace> expectedEval("Expected output", N_sigmoid);
            for(int j = 0; j < N_sigmoid; j++) {
                double pt_j = centers(j);
                evalPts(0,j+2) = pt_j;
                double exp_eval_j = 0.;
                for(int k = 0; k < N_sigmoid; k++) {
                    exp_eval_j += coeffs(k)*TestType::Evaluate(widths(k)*(pt_j - centers(k)));
                }
                expectedEval(j) = exp_eval_j;
            }
            StridedMatrix<double,MemorySpace> out = Sigmoid.Evaluate(evalPts);
            double approxTol = 1e-5;
            REQUIRE_THAT(out(0,0), Matchers::WithinAbs(0.0      , approxTol));
            REQUIRE_THAT(out(0,1), Matchers::WithinAbs(coeff_sum, approxTol));
            for(int j = 0; j < N_sigmoid; j++) {
                REQUIRE_THAT(out(0,j+2), Matchers::WithinAbs(expectedEval(j), approxTol));
            }
        }
        TestSigmoidGradients(Sigmoid, N_grad_points, fd_delta, coeffs);
    }
}