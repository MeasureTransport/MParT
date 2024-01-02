#include <Kokkos_Core.hpp>

namespace mpart {
enum BasisHomogeneity {Homogeneous, OffdiagHomogeneous, Heterogeneous};

template<BasisHomogeneity HowHomogeneous, typename BasisEvaluatorType>
struct BasisEvaluator {
    // dimension size, Basis evaluator
    BasisEvaluator(int, BasisEvaluatorType) {
        assert(false); // TODO: Figure this out
    }
    // EvaluateAll(dim, output, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateAll(int, double*, int, double) const {
        assert(false);
    }
    // EvaluateDerivatives(dim, output_eval, output_deriv, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(int, double*, double*, int, double) const {
        assert(false);
    }
    // EvaluateSecondDerivatives(dim, output_eval, output_deriv, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(int, double*, double*, double*, int, double) const {
        assert(false);
    }
#if defined(MPART_HAS_CEREAL)
    template<typename Archive>
    void serialize(Archive &ar) {
        assert(false);
    }
#endif
};

template<typename BasisEvaluatorType>
struct BasisEvaluator<BasisHomogeneity::Homogeneous, BasisEvaluatorType> {
    // dimension size, Basis evaluator
    BasisEvaluator(int, BasisEvaluatorType const& basis1d): basis1d_(basis1d) {}
    template<typename... Args>
    BasisEvaluator(Args... args): basis1d_(args...) {}
    // EvaluateAll(dim, output, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateAll(int, double* output, int max_order, double input) const {
        basis1d_.EvaluateAll(output, max_order, input);
    }
    // EvaluateDerivatives(dim, output_eval, output_deriv, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(int, double* output, double* output_diff, int max_order, double input) const {
        basis1d_.EvaluateDerivatives(output, output_diff, max_order, input);
    }
    // EvaluateSecondDerivatives(dim, output_eval, output_deriv, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(int, double* output, double* output_diff, double* output_diff2, int max_order, double input) const {
        basis1d_.EvaluateSecondDerivatives(output, output_diff, output_diff2, max_order, input);
    }
#if defined(MPART_HAS_CEREAL)
    template<typename Archive>
    void serialize(Archive &ar) {
        ar(basis1d_);
    }
#endif
    BasisEvaluatorType basis1d_;
};


template<typename BasisEvaluatorType1, typename BasisEvaluatorType2>
struct BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<BasisEvaluatorType1, BasisEvaluatorType2>> {
    
    BasisEvaluator(int dim, Kokkos::pair<BasisEvaluatorType1, BasisEvaluatorType2> const& basis1d): offdiag_(basis1d.first), diag_(basis1d.second), dim_(dim) {}
    BasisEvaluator(int dim, BasisEvaluatorType1 const &offdiag, BasisEvaluatorType2 const &diag): offdiag_(offdiag), diag_(diag), dim_(dim) {}
    // EvaluateAll(dim, output, max_order, input)
    // dim is zero-based indexing
    KOKKOS_INLINE_FUNCTION void EvaluateAll(int dim, double* output, int max_order, double input) const {
        if(dim < dim_-1) offdiag_.EvaluateAll(output, max_order, input);
        else diag_.EvaluateAll(output, max_order, input);
    }
    // EvaluateDerivatives(dim, output_eval, output_deriv, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(int dim, double* output, double* output_diff, int max_order, double input) const {
        if(dim < dim_-1) offdiag_.EvaluateDerivatives(output, output_diff, max_order, input);
        else diag_.EvaluateDerivatives(output, output_diff, max_order, input);
    }
    // EvaluateSecondDerivatives(dim, output_eval, output_deriv, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(int dim, double* output, double* output_diff, double* output_diff2, int max_order, double input) const {
        if(dim < dim_-1) offdiag_.EvaluateSecondDerivatives(output, output_diff, output_diff2, max_order, input);
        else diag_.EvaluateSecondDerivatives(output, output_diff, output_diff2, max_order, input);
    }
#if defined(MPART_HAS_CEREAL)
    template<typename Archive>
    void serialize(Archive &ar) {
        ar(dim_, offdiag_, diag_);
    }
#endif
    int dim_;
    BasisEvaluatorType1 offdiag_;
    BasisEvaluatorType2 diag_;
};

/// @brief Type to represent a basis evaluation when we use different basis functions for different variables
/// @tparam CommonBasisEvaluatorType Some type all the basis functions fall under
template<typename CommonBasisEvaluatorType>
struct BasisEvaluator<BasisHomogeneity::Heterogeneous, std::vector<std::shared_ptr<CommonBasisEvaluatorType>>> {
    BasisEvaluator(int dim, std::vector<std::shared_ptr<CommonBasisEvaluatorType>> const& basis1d): basis1d_(basis1d) {
        assert(dim == basis1d.size()); // TODO: Fix
    }
    // EvaluateAll(dim, output, max_order, input)
    // dim is zero-based indexing
    KOKKOS_INLINE_FUNCTION void EvaluateAll(int dim, double* output, int max_order, double input) const {
        basis1d_[dim]->EvaluateAll(output, max_order, input);
    }
    // EvaluateDerivatives(dim, output_eval, output_deriv, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(int dim, double* output, double* output_diff, int max_order, double input) const{
        basis1d_[dim]->EvaluateDerivatives(output, output_diff, max_order, input);
    }
    // EvaluateSecondDerivatives(dim, output_eval, output_deriv, max_order, input)
    KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(int dim, double* output, double* output_diff, double* output_diff2, int max_order, double input) const {
        basis1d_[dim]->EvaluateSecondDerivatives(output, output_diff, output_diff2, max_order, input);
    }
#if defined(MPART_HAS_CEREAL)
    template<typename Archive>
    void serialize(Archive &ar) {
        ar(basis1d_);
    }
#endif
    std::vector<std::shared_ptr<CommonBasisEvaluatorType>> basis1d_;
    // NOTE: shared_ptr is necessary to avoid type slicing
};
} // namespace mpart