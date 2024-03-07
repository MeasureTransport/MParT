#include <catch2/catch_all.hpp>
#include "MParT/Sigmoid.h"
#include <iostream>

using namespace mpart;
using namespace Catch;
using namespace Catch::Matchers;

void CheckNearZero(double calc, double ref, double delta=1e-12, double tol=1e-12) {
    if(ref == 0.) REQUIRE_THAT(calc, WithinAbs(0., tol));
    else REQUIRE_THAT(calc, WithinRel(ref, delta));
}

template<typename MemorySpace, typename Function>
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
    Kokkos::View<double*, MemorySpace> output_d ("Output", N_grad_points*(max_order+1));
    Kokkos::View<double*, MemorySpace> output_pos_fd_d ("Output", N_grad_points*(max_order+1));
    Kokkos::View<double*, MemorySpace> output_deriv_d ("Output", N_grad_points*(max_order+1));
    Kokkos::View<double*, MemorySpace> output_deriv_pos_fd_d ("Output", N_grad_points*(max_order+1));
    Kokkos::View<double*, MemorySpace> output_2deriv_d ("Output", N_grad_points*(max_order+1));
    Kokkos::View<double*, MemorySpace> output_diff_d ("Output", N_grad_points*(max_order+1));
    Kokkos::View<double*, MemorySpace> output_diff_pos_fd_d ("Output", N_grad_points*(max_order+1));
    Kokkos::View<double*, MemorySpace> output_diff_2deriv_d ("Output", N_grad_points*(max_order+1));
    Kokkos::View<double*, MemorySpace> output_diff2_d ("Output", N_grad_points*(max_order+1));
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(unsigned int i) {
        unsigned int offset = i*(max_order+1);
        Sigmoid.EvaluateAll(output_d.data() + offset, max_order, gradPts(i));
        Sigmoid.EvaluateAll(output_pos_fd_d.data() + offset, max_order, gradPts_plus_delta(i));
        Sigmoid.EvaluateDerivatives(output_deriv_d.data() + offset, output_diff_d.data() + offset, max_order, gradPts(i));
        Sigmoid.EvaluateDerivatives(output_deriv_pos_fd_d.data() + offset, output_diff_pos_fd_d.data() + offset, max_order, gradPts_plus_delta(i));
        Sigmoid.EvaluateSecondDerivatives(output_2deriv_d.data() + offset, output_diff_2deriv_d.data() + offset, output_diff2_d.data() + offset, max_order, gradPts(i));
    });
    Kokkos::fence();
    Kokkos::View<double*, Kokkos::HostSpace> output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_d);
    Kokkos::View<double*, Kokkos::HostSpace> output_pos_fd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_pos_fd_d);
    Kokkos::View<double*, Kokkos::HostSpace> output_deriv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_deriv_d);
    Kokkos::View<double*, Kokkos::HostSpace> output_deriv_pos_fd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_deriv_pos_fd_d);
    Kokkos::View<double*, Kokkos::HostSpace> output_2deriv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_2deriv_d);
    Kokkos::View<double*, Kokkos::HostSpace> output_diff = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_diff_d);
    Kokkos::View<double*, Kokkos::HostSpace> output_diff_pos_fd = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_diff_pos_fd_d);
    Kokkos::View<double*, Kokkos::HostSpace> output_diff_2deriv = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_diff_2deriv_d);
    Kokkos::View<double*, Kokkos::HostSpace> output_diff2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_diff2_d);

    for(int i = 0; i < N_grad_points; i++) {
        for(int j = 0; j <= max_order; j++) {
            double fd_diff = (output_pos_fd(j)-output(j))/fd_delta;
            double fd_diff2 = (output_diff_pos_fd(j) - output_diff(j))/fd_delta;
            CheckNearZero(output_deriv(j), output(j), 1e-12);
            CheckNearZero(output_2deriv(j), output(j), 1e-12);
            CheckNearZero(output_deriv_pos_fd(j), output_pos_fd(j), 1e-12);
            CheckNearZero(output_diff(j), fd_diff, sqrt(fd_delta));
            CheckNearZero(output_diff_2deriv(j), fd_diff, sqrt(fd_delta));
            CheckNearZero(output_diff2(j), fd_diff2, sqrt(fd_delta));
        }
    }
}

using TestType1 = std::pair< SigmoidTypeSpace::Logistic, std::integral_constant<int, 0> >;

#if defined(MPART_ENABLE_GPU)
using TestType2 = std::pair< SigmoidTypeSpace::Logistic, std::integral_constant<int, 1> >;
#else
using TestType2 = std::pair< SigmoidTypeSpace::Logistic, std::integral_constant<int, -1> >;
#endif

TEMPLATE_TEST_CASE("Sigmoid1d","[sigmoid1d]", TestType1, TestType2) {
if (TestType::second_type::value >= 0) { // Don't worry about testing the host twice

    using SigmoidType = typename TestType::first_type;
    constexpr bool is_device = (TestType::second_type::value == 1);
    using MemorySpace = std::conditional_t< is_device, DeviceSpace, Kokkos::HostSpace>;
    using ExecutionSpace = typename MemoryToExecution<MemorySpace>::Space;
    SECTION("Initialization") {
        Kokkos::View<double*, MemorySpace> centers("Sigmoid Centers", 2);
        Kokkos::View<double*, MemorySpace> widths("Sigmoid Widths", 3);
        Kokkos::View<double*, MemorySpace> weights("Sigmoid weights", 2);

        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,SigmoidType>(centers, widths, weights)), std::invalid_argument);
        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,SigmoidType>(centers, widths)), std::invalid_argument);
        int N_wrong = 2+1+2+3+5;
        centers = Kokkos::View<double*, MemorySpace>("Sigmoid Centers", N_wrong);
        widths = Kokkos::View<double*, MemorySpace>("Sigmoid widths", N_wrong);
        weights = Kokkos::View<double*, MemorySpace>("Sigmoid weights", N_wrong);
        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,SigmoidType>(centers, widths, weights)), std::invalid_argument);
        CHECK_THROWS_AS((Sigmoid1d<MemorySpace,SigmoidType>(centers, widths)), std::invalid_argument);
        int N_wrong_arr[4] = {0, 1, 2+(1+3)};
        for(int N_wrong : N_wrong_arr) {
            centers = Kokkos::View<double*, MemorySpace>("Sigmoid Centers", N_wrong);
            widths = Kokkos::View<double*, MemorySpace>("Sigmoid widths", N_wrong);
            weights = Kokkos::View<double*, MemorySpace>("Sigmoid weights", N_wrong);
            CHECK_THROWS_AS((Sigmoid1d<MemorySpace,SigmoidType>(centers, widths, weights)), std::invalid_argument);
            CHECK_THROWS_AS((Sigmoid1d<MemorySpace,SigmoidType>(centers, widths)), std::invalid_argument);
        }
        centers = Kokkos::View<double*, MemorySpace>("Sigmoid Centers", 2);
        widths = Kokkos::View<double*, MemorySpace>("Sigmoid widths", 2);
        Sigmoid1d<MemorySpace,SigmoidType> Sigmoid {centers, widths};
        CHECK(Sigmoid.GetOrder() == 1 + 2); // Affine + 2 edge terms
        centers = Kokkos::View<double*, MemorySpace>("Sigmoid Centers", 3);
        widths = Kokkos::View<double*, MemorySpace>("Sigmoid widths", 3);
        Sigmoid = Sigmoid1d<MemorySpace,SigmoidType>(centers, widths);
        CHECK(Sigmoid.GetOrder() == 1 + 2 + 1); // Affine + 2 edge terms + 1 sigmoid
    }
    Kokkos::fence();

    const double support_bound = 100.;

    SECTION("Single Sigmoid") {
        const int num_sigmoid = 1;
        const int order = num_sigmoid+1+2;
        const int param_length = 2 + num_sigmoid*(num_sigmoid+1)/2;
        Kokkos::View<double*, MemorySpace> center("Sigmoid Center", param_length);
        Kokkos::View<double*, MemorySpace> width("Sigmoid Width", param_length);
        Kokkos::View<double*, MemorySpace> weight("Sigmoid Weight", param_length);
        Kokkos::deep_copy(center, 0.);
        Kokkos::deep_copy(width, 1.);
        Kokkos::deep_copy(weight, 1.);

        Sigmoid1d<MemorySpace,SigmoidType> Sigmoid (center, width, weight);

        // Ensure the sigmoid f is monotone over a grid, f(-infty) = 0, f(+infty) = 1, f(0) = 0.5
        const int num_pts_grid = 100;
        const double grid_bdry = 5.;

        Kokkos::View<double*, MemorySpace> eval_pts_d {"Grid points device", num_pts_grid+3};
        Kokkos::RangePolicy<ExecutionSpace> pts_policy {0, num_pts_grid};
        Kokkos::parallel_for(pts_policy, KOKKOS_LAMBDA(unsigned int p){
            if(p == 0) {
                eval_pts_d(0) = -support_bound;
                eval_pts_d(1) = 0.;
                eval_pts_d(2) = support_bound;
            }
            eval_pts_d(p+3) = -grid_bdry + p*2*grid_bdry/(num_pts_grid-1);
        });
        auto eval_pts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eval_pts_d);
        Kokkos::fence();

        double expect_output[(order+1)*3] = {
            1., -support_bound, -support_bound,            0., 0.0,
            1.,             0.,   -std::log(2),   std::log(2), 0.5,
            1.,  support_bound,             0., support_bound, 1.0};

        Kokkos::View<double*, MemorySpace> output_d {"Output storage device", (order+1)*(num_pts_grid+3)};
        Kokkos::RangePolicy<ExecutionSpace> eval_policy {0, num_pts_grid+3};
        Kokkos::parallel_for(eval_policy, KOKKOS_LAMBDA(unsigned int j){
            Sigmoid.EvaluateAll(output_d.data() + j*(order+1), order, eval_pts_d(j));
        });
        Kokkos::fence();
        Kokkos::View<double*, Kokkos::HostSpace> output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_d);

        int j = 0;
        for(; j < 3; j++) {
            for(int i = 0; i < (order+1); i++) {
                int idx = j*(order+1)+i;
                REQUIRE_THAT(output(idx), WithinRel(expect_output[idx], 1e-12));
            }
        }

        double prev = 0.;
        for(; j < num_pts_grid + 3; j++) {
            CHECK(output(j*(order+1)  ) == 1.);
            CHECK(output(j*(order+1)+1) == eval_pts(j));
            double next = output(j*(order+1)+3);
            CHECK(next > prev);
            prev = next;
        }
        TestSigmoidGradients<MemorySpace>(Sigmoid, 100, 1e-7);
    }

    SECTION("Multiple Sigmoids") {
        const int num_sigmoids = 3;
        const int order = num_sigmoids+1+2;
        const int param_length = 2 + num_sigmoids*(num_sigmoids+1)/2;
        Kokkos::View<double*, Kokkos::HostSpace> centers("Sigmoid centers", param_length);
        Kokkos::View<double*, Kokkos::HostSpace> widths("Sigmoid widths", param_length);
        Kokkos::View<double*, Kokkos::HostSpace> weights("Sigmoid weights", param_length);
        double edge_bound = 3.;
        centers(0) = -edge_bound; widths(0) = 2*edge_bound/10; weights(0) = 1.;
        centers(1) =  edge_bound; widths(1) = 2*edge_bound/10; weights(1) = 1.;
        int param_idx = 2;
        for(int curr_order = 1; curr_order <= num_sigmoids; curr_order++) {
            for(int i = 0; i < curr_order; i++) {
                centers(param_idx) = 4*(-(curr_order-1)/2 + i);
                widths(param_idx) = 1/((double)i+1);
                weights(param_idx) = 1./curr_order;
                param_idx++;
            }
        }
        auto centers_d = Kokkos::create_mirror_view_and_copy(MemorySpace(), centers);
        auto widths_d = Kokkos::create_mirror_view_and_copy(MemorySpace(), widths);
        auto weights_d = Kokkos::create_mirror_view_and_copy(MemorySpace(), weights);
        Sigmoid1d<MemorySpace,SigmoidType> Sigmoid {centers_d, widths_d, weights_d};

        // Ensure the sigmoid f is monotone over a grid, f(-infty) = 0, f(+infty) = 1
        const int num_pts_grid = 100;
        const double grid_bdry = 5.;
        Kokkos::View<double*,MemorySpace> eval_pts_d {"Grid points", num_pts_grid+2};
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, num_pts_grid), KOKKOS_LAMBDA(int p) {
            if(p == 0) {
                eval_pts_d(0) = -support_bound;
                eval_pts_d(1) = support_bound;
            }
            eval_pts_d(p+2) = -grid_bdry + 2*p*grid_bdry/(num_pts_grid-1);
        });
        auto eval_pts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eval_pts_d);
        Kokkos::View<double*, MemorySpace> output_d {"Output storage device", (order+1)*(num_pts_grid+2)};
        Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, num_pts_grid+2), KOKKOS_LAMBDA(int j) {
            Sigmoid.EvaluateAll(output_d.data() + j*(order+1), order, eval_pts_d(j));
        });
        Kokkos::fence();
        auto output = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), output_d);

        double expect_output[5*2] = {
            1., -support_bound, -(support_bound-edge_bound)*widths(0), 0.0, 0.,
            1.,  support_bound,  0.,  (support_bound-edge_bound)*widths(1), 1.
        };

        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 4; j++) {
                CHECK(output(i*(order+1)+j) == expect_output[i*5 + j]);
            }
            for(int j = 4; j <= order; j++) {
                REQUIRE_THAT(output(i*(order+1)+j), WithinAbs(expect_output[i*5+4], 1e-12));
            }
        }
        unsigned int output_start_idx = 2*(order+1);
        double prev[order+1] = {0.};
        // set prev for left edge term to negative infty
        prev[2] = -2*support_bound;
        for(int j = 0; j < num_pts_grid; j++) {
            CHECK(output(output_start_idx + 0) == 1.); // Constant
            CHECK(output(output_start_idx + 1) == eval_pts(j+2)); // Linear
            for(int curr_sigmoid = 2; curr_sigmoid <= order; curr_sigmoid++) { // edge+sigmoids
                double out_val = output(output_start_idx + curr_sigmoid);
                CHECK(out_val > prev[curr_sigmoid]);
                prev[curr_sigmoid] = out_val;
            }
            output_start_idx += order+1;
        }
        TestSigmoidGradients<MemorySpace>(Sigmoid, 100, 1e-7);
    }
}}