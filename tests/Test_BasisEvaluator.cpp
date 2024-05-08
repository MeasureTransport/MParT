#include "MParT/BasisEvaluator.h"
#include <catch2/catch_all.hpp>

using namespace mpart;
using namespace Catch;

struct IdentityEvaluator{
    KOKKOS_INLINE_FUNCTION void EvaluateAll(double* output, int max_order, double input) const {
        for(int i = 0; i <= max_order; i++) {
            output[i] = input;
        }
    }
    KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(double* output, double* output_diff, int max_order, double input) const {
        for(int i = 0; i <= max_order; i++) {
            output[i] = input;
            output_diff[i] = 1;
        }
    }
    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(double* output, double* output_diff, double* output_diff2, int max_order, double input) const {
        for(int i = 0; i <= max_order; i++) {
            output[i] = input;
            output_diff[i] = 1;
            output_diff2[i] = 0;
        }
    }
};

struct NegativeEvaluator {
    KOKKOS_INLINE_FUNCTION void EvaluateAll(double* output, int max_order, double input) const {
        for(int i = 0; i <= max_order; i++) {
            output[i] = -input;
        }
    }
    KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(double* output, double* output_diff, int max_order, double input) const {
        for(int i = 0; i <= max_order; i++) {
            output[i] = -input;
            output_diff[i] = -1;
        }
    }
    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(double* output, double* output_diff, double* output_diff2, int max_order, double input) const {
        for(int i = 0; i <= max_order; i++) {
            output[i] = -input;
            output_diff[i] = -1;
            output_diff2[i] = 0;
        }
    }
};

TEST_CASE( "Testing basis evaluators", "[BasisEvaluators]") {
    unsigned int dim = 3;
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
        using Eval_T = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<NegativeEvaluator, IdentityEvaluator>, Identity>;
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
        dim = 8;
        auto eval = CreateHeterogeneousBasisEvaluator<Identity>(3, 4, 1, IdentityEvaluator(), NegativeEvaluator(), IdentityEvaluator());
        for(int d = 0; d < dim; d++) {
            reset_out();
            eval.EvaluateAll(d, out, max_order, point);
            bool isNegative = d >= 3 && d < 7;
            double expectedEval = isNegative ? -point : point;
            for(int i = 0; i <= max_order; i++) REQUIRE(out[i] == expectedEval);
            
            reset_out();
            eval.EvaluateDerivatives(d, out, out_diff, max_order, point);
            for(int i = 0; i <= max_order; i++) {
                bool is_id_valid = out[i] == (expectedEval);
                is_id_valid &= out_diff[i] == (isNegative ? -1: 1);
                CHECK(is_id_valid);
            }

            reset_out();
            eval.EvaluateSecondDerivatives(d, out, out_diff, out_diff2, max_order, point);
            for(int i = 0; i <= max_order; i++) {
                bool is_id_valid = out[i] == (expectedEval);
                is_id_valid &= out_diff[i] == (isNegative ? -1 : 1);
                is_id_valid &= out_diff2[i] == 0.;
                CHECK(is_id_valid);
            }
        }
    }
}
