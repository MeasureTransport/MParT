#include <catch2/catch_all.hpp>
#include "MParT/Sigmoid.h"

using namespace mpart;
using namespace Catch;
using namespace Catch::Matchers;
using MemorySpace = Kokkos::HostSpace;

template<typename Function>
void TestSigmoidGradients(Function Sigmoid, unsigned int N_grad_points, double fd_delta) {
    Kokkos::View<double*, MemorySpace> gradPts("Gradient points", N_grad_points);
    Kokkos::View<double*, MemorySpace> gradPts_plus_delta("Gradient points plus delta", N_grad_points);
    Kokkos::parallel_for(N_grad_points, KOKKOS_LAMBDA(unsigned int point_index) {
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
            double fd_diff2 = (output_deriv_pos_fd[j] - output_deriv[j])/fd_delta;
            REQUIRE_THAT(output_deriv[j], WithinRel(output[j], 1e-12));
            REQUIRE_THAT(output_2deriv[j], WithinRel(output[j], 1e-12));
            REQUIRE_THAT(output_deriv_pos_fd[j], WithinRel(output_pos_fd[j], 1e-12));
            REQUIRE_THAT(output_diff[j], WithinRel(fd_diff, 10*fd_delta));
            REQUIRE_THAT(output_diff_2deriv[j], WithinRel(fd_diff, 10*fd_delta));
            REQUIRE_THAT(output_diff2[j], WithinRel(fd_diff2, 10*fd_delta));
        }
    }
}

TEMPLATE_TEST_CASE("Sigmoid1d","[sigmoid1d]", SigmoidTypes::Logistic) {
    SECTION("Initialization") {
        Kokkos::View<double*, MemorySpace> centers("Sigmoid Centers", 2);
        Kokkos::View<double*, MemorySpace> widths("Sigmoid Widths", 1);
        Kokkos::View<double*, MemorySpace> weights("Sigmoid weights", 1);

        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths, weights)), std::invalid_argument);
        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths)), std::invalid_argument);
        int N_wrong = 1+2+3+5;
        centers = Kokkos::View<double*, MemorySpace>("Sigmoid Centers", N_wrong);
        widths = Kokkos::View<double*, MemorySpace>("Sigmoid widths", N_wrong);
        weights = Kokkos::View<double*, MemorySpace>("Sigmoid weights", N_wrong);
        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths, weights)), std::invalid_argument);
        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,TestType>(centers, widths)), std::invalid_argument);
    }

    unsigned int N_grad_points = 100;
    double fd_delta = 1e-6;
    const double support_bound = 100.;

    SECTION("Single Sigmoid") {
        const int order = 1;
        const int param_length = order*(order+1)/2;
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
        for(int p = 0; p < num_pts_grid; p++) eval_pts[p+3] = -grid_bdry + 2*(p-3)*grid_bdry/num_pts_grid;
        double expect_output[(order+1)*3] = {0., 0., 0., 0.5, 0., 1.};
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
            // SPECIFIC FOR order==1
            CHECK(output[j*2] == 0.);
            double next = output[j*2+1];
            CHECK(next > prev);
            prev = next;
        }
        TestSigmoidGradients(Sigmoid, 100, 1e-7);
    }

    SECTION("Multiple Sigmoids") {
        const int order = 3;
        const int param_length = order*(order+1)/2;
        Kokkos::View<double*, MemorySpace> center("Sigmoid Center", param_length);
        Kokkos::View<double*, MemorySpace> width("Sigmoid Width", param_length);
        Kokkos::View<double*, MemorySpace> weight("Sigmoid Weight", param_length);
        int param_idx = 0;
        for(int curr_order = 1; curr_order <= order; curr_order++) {
            for(int i = 0; i < curr_order; i++) {
                center(param_idx) = -(curr_order-1)/2 + i;
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
            eval_pts[p] = -grid_bdry + 2*p*grid_bdry/num_pts_grid;
        }
        double output[order+1];
        Sigmoid.EvaluateAll(output, order, -support_bound);
        CHECK(output[0] == 0.);
        for(int i = 1; i <= order; i++) REQUIRE_THAT(output[i], WithinAbs(0., 1e-12));
        Sigmoid.EvaluateAll(output, order,  support_bound);
        CHECK(output[0] == 0.);
        for(int i = 1; i <= order; i++) REQUIRE_THAT(output[i], WithinAbs(1., 1e-12));
        double prev[order+1] = {0., 0., 0., 0.};
        for(int j = 0; j < num_pts_grid; j++) {
            Sigmoid.EvaluateAll(output, order, eval_pts[j]);
            CHECK(output[0] == 0.);
            for(int curr_order = 1; curr_order <= order; curr_order++) {
                CHECK(output[curr_order] > prev[curr_order]);
                prev[curr_order] = output[curr_order];
            }
        }
        TestSigmoidGradients(Sigmoid, 100, 1e-7);
    }
}