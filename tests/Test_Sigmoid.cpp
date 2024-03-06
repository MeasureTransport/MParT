#include <catch2/catch_all.hpp>
#include "MParT/Sigmoid.h"

using namespace mpart;
using namespace Catch;
using namespace Catch::Matchers;
using MemorySpace = Kokkos::HostSpace;

void CheckNearZero(double calc, double ref, double delta=1e-12, double tol=1e-12) {
    if(ref == 0.) REQUIRE_THAT(calc, WithinAbs(0., tol));
    else REQUIRE_THAT(calc, WithinRel(ref, delta));
}

template<typename Function>
void TestSigmoidGradients(Function Sigmoid, unsigned int N_grad_points, double fd_delta) {
    Kokkos::View<double*, MemorySpace> gradPts("Gradient points", N_grad_points);
    Kokkos::View<double*, MemorySpace> gradPts_plus_delta("Gradient points plus delta", N_grad_points);
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy {0u, N_grad_points};
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(unsigned int point_index) {
        double gradPt = 3.0*(-1.0 + 2*((double) point_index)/((double) N_grad_points-1));
        gradPts(point_index) = gradPt;
        gradPts_plus_delta(point_index) = gradPt + fd_delta;
    });
    int max_order = Sigmoid.GetOrder();
    // Create output array for each possible evaluation
    double output[max_order+1];
    double output_pos_fd[max_order+1];
    double output_deriv[max_order+1];
    double output_deriv_pos_fd[max_order+1];
    double output_2deriv[max_order+1];
    double output_diff[max_order+1];
    double output_diff_pos_fd[max_order+1];
    double output_diff_2deriv[max_order+1];
    double output_diff2[max_order+1];
    for(int i = 0; i < N_grad_points; i++) {
        Sigmoid.EvaluateAll(output, max_order, gradPts(i));
        Sigmoid.EvaluateAll(output_pos_fd, max_order, gradPts_plus_delta(i));
        Sigmoid.EvaluateDerivatives(output_deriv, output_diff, max_order, gradPts(i));
        Sigmoid.EvaluateDerivatives(output_deriv_pos_fd, output_diff_pos_fd, max_order, gradPts_plus_delta(i));
        Sigmoid.EvaluateSecondDerivatives(output_2deriv, output_diff_2deriv, output_diff2, max_order, gradPts(i));
        for(int j = 0; j <= max_order; j++) {
            double fd_diff = (output_pos_fd[j]-output[j])/fd_delta;
            double fd_diff2 = (output_diff_pos_fd[j] - output_diff[j])/fd_delta;
            CheckNearZero(output_deriv[j], output[j], 1e-12);
            CheckNearZero(output_2deriv[j], output[j], 1e-12);
            CheckNearZero(output_deriv_pos_fd[j], output_pos_fd[j], 1e-12);
            CheckNearZero(output_diff[j], fd_diff, sqrt(fd_delta));
            CheckNearZero(output_diff_2deriv[j], fd_diff, sqrt(fd_delta));
            CheckNearZero(output_diff2[j], fd_diff2, sqrt(fd_delta));
        }
    }
}

TEMPLATE_TEST_CASE("Sigmoid1d","[sigmoid1d]", SigmoidTypeSpace::Logistic) {
    SECTION("Initialization") {
        Kokkos::View<double*, MemorySpace> centers("Sigmoid Centers", 2);
        Kokkos::View<double*, MemorySpace> widths("Sigmoid Widths", 3);
        Kokkos::View<double*, MemorySpace> weights("Sigmoid weights", 2);

        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths, weights)), std::invalid_argument);
        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths)), std::invalid_argument);
        int N_wrong = 2+1+2+3+5;
        centers = Kokkos::View<double*, MemorySpace>("Sigmoid Centers", N_wrong);
        widths = Kokkos::View<double*, MemorySpace>("Sigmoid widths", N_wrong);
        weights = Kokkos::View<double*, MemorySpace>("Sigmoid weights", N_wrong);
        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths, weights)), std::invalid_argument);
        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths)), std::invalid_argument);
        int N_wrong_arr[4] = {0, 1, 2+(1+3)};
        for(int N_wrong : N_wrong_arr) {
            centers = Kokkos::View<double*, MemorySpace>("Sigmoid Centers", N_wrong);
            widths = Kokkos::View<double*, MemorySpace>("Sigmoid widths", N_wrong);
            weights = Kokkos::View<double*, MemorySpace>("Sigmoid weights", N_wrong);
            CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths, weights)), std::invalid_argument);
            CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths)), std::invalid_argument);
        }
        centers = Kokkos::View<double*, MemorySpace>("Sigmoid Centers", 2);
        widths = Kokkos::View<double*, MemorySpace>("Sigmoid widths", 2);
        Sigmoid1d<MemorySpace,TestType> Sigmoid = Sigmoid1d<MemorySpace,TestType>(centers, widths);
        CHECK(Sigmoid.GetOrder() == 1 + 2); // Affine + 2 edge terms
        centers = Kokkos::View<double*, MemorySpace>("Sigmoid Centers", 3);
        widths = Kokkos::View<double*, MemorySpace>("Sigmoid widths", 3);
        Sigmoid = Sigmoid1d<MemorySpace,TestType>(centers, widths);
        CHECK(Sigmoid.GetOrder() == 1 + 2 + 1); // Affine + 2 edge terms + 1 sigmoid
    }

    const double support_bound = 100.;

    SECTION("Single Sigmoid") {
        const int num_sigmoid = 1;
        const int order = num_sigmoid+1+2;
        const int param_length = 2 + num_sigmoid*(num_sigmoid+1)/2;
        Kokkos::View<double*, MemorySpace> center("Sigmoid Center", param_length);
        Kokkos::View<double*, MemorySpace> width("Sigmoid Width", param_length);
        Kokkos::View<double*, MemorySpace> weight("Sigmoid Weight", param_length);
        for(int i = 0; i < param_length; i++) {
            center(i) = 0; width(i) = 1; weight(i) = 1.;
        }
        Sigmoid1d<MemorySpace,TestType> Sigmoid (center, width, weight);

        // Ensure the sigmoid f is monotone over a grid, f(-infty) = 0, f(+infty) = 1, f(0) = 0.5
        const int num_pts_grid = 100;
        const double grid_bdry = 5.;
        double eval_pts[num_pts_grid+3];
        eval_pts[0] = -support_bound; eval_pts[1] = 0.; eval_pts[2] = support_bound;
        for(int p = 0; p < num_pts_grid; p++) eval_pts[p+3] = -grid_bdry + p*2*grid_bdry/(num_pts_grid-1);
        double expect_output[(order+1)*3] = {
            1., eval_pts[0], eval_pts[0], 0., 0.,
            1., eval_pts[1], -std::log(2), std::log(2), 0.5,
            1., eval_pts[2], 0., eval_pts[2], 1.};
        double output[(order+1)*(num_pts_grid+3)];
        
        int j = 0;
        for(; j < 3; j++) {
            Sigmoid.EvaluateAll(output+j*(order+1), order, eval_pts[j]);
            for(int i = 0; i < (order+1); i++) {
                int idx = j*(order+1)+i;
                REQUIRE_THAT(output[idx], WithinRel(expect_output[idx], 1e-12));
            }
        }
        double prev = 0.;
        for(; j < num_pts_grid + 3; j++) {
            Sigmoid.EvaluateAll(output+j*(order+1), order, eval_pts[j]);
            CHECK(output[j*(order+1)  ] == 1.);
            CHECK(output[j*(order+1)+1] == eval_pts[j]);
            double next = output[j*(order+1)+3];
            CHECK(next > prev);
            prev = next;
        }
        TestSigmoidGradients(Sigmoid, 100, 1e-7);
    }

    SECTION("Multiple Sigmoids") {
        const int num_sigmoids = 3;
        const int order = num_sigmoids+1+2;
        const int param_length = 2 + num_sigmoids*(num_sigmoids+1)/2;
        Kokkos::View<double*, MemorySpace> center("Sigmoid Center", param_length);
        Kokkos::View<double*, MemorySpace> width("Sigmoid Width", param_length);
        Kokkos::View<double*, MemorySpace> weight("Sigmoid Weight", param_length);
        double edge_bound = 3.;
        center(0) = -edge_bound; width(0) = 2*edge_bound/10; weight(0) = 1.;
        center(1) =  edge_bound; width(1) = 2*edge_bound/10; weight(1) = 1.;
        int param_idx = 2;
        for(int curr_order = 1; curr_order <= num_sigmoids; curr_order++) {
            for(int i = 0; i < curr_order; i++) {
                center(param_idx) = 4*(-(curr_order-1)/2 + i);
                width(param_idx) = 1/((double)i+1);
                weight(param_idx) = 1./curr_order;
                param_idx++;
            }
        }
        Sigmoid1d<MemorySpace,TestType> Sigmoid (center, width, weight);

        // Ensure the sigmoid f is monotone over a grid, f(-infty) = 0, f(+infty) = 1
        const int num_pts_grid = 100;
        const double grid_bdry = 5.;
        double eval_pts[num_pts_grid];
        for(int p = 0; p < num_pts_grid; p++) {
            eval_pts[p] = -grid_bdry + 2*p*grid_bdry/(num_pts_grid-1);
        }
        double output[order+1];
        Sigmoid.EvaluateAll(output, order, -support_bound);
        CHECK(output[0] == 1.);
        CHECK(output[1] == -support_bound);
        CHECK(output[2] == -(support_bound-edge_bound)*width(0));
        CHECK(output[3] == 0.);
        for(int i = 4; i <= order; i++) {
            REQUIRE_THAT(output[i], WithinAbs(0., 1e-12));
        }
        Sigmoid.EvaluateAll(output, order,  support_bound);
        CHECK(output[0] == 1.);
        CHECK(output[1] == support_bound);
        CHECK(output[2] == 0.);
        CHECK(output[3] == (support_bound-edge_bound)*width(1));
        for(int i = 4; i <= order; i++) REQUIRE_THAT(output[i], WithinAbs(1., 1e-12));
        double prev[order+1] = {0.};
        // set prev for left edge term to negative infty
        prev[2] = -2*support_bound;
        for(int j = 0; j < num_pts_grid; j++) {
            Sigmoid.EvaluateAll(output, order, eval_pts[j]);
            CHECK(output[0] == 1.);
            CHECK(output[1] == eval_pts[j]);
            for(int curr_sigmoid = 2; curr_sigmoid <= order; curr_sigmoid++) {
                CHECK(output[curr_sigmoid] > prev[curr_sigmoid]);
                prev[curr_sigmoid] = output[curr_sigmoid];
            }
        }
        TestSigmoidGradients(Sigmoid, 100, 1e-7);
    }
}