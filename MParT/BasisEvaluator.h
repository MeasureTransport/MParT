#include <Kokkos_Core.hpp>
#include "MParT/PositiveBijectors.h"

namespace mpart {
/**
 * @brief Flags for controlling how "homogeneous/heterogeneous" a real-valued
 * multivariate basis function is
 *
 * Assuming you want create a function
 * \f$f_{\vec{\alpha}}:\mathbb{R}^{d+1}\to\mathbb{R}\f$ which has multi-index
 * \f$\vec{\alpha}\f$, there are a few possibilities:
 *
 * - All diagonal and offdiagonal univariate functions are identical, i.e.
 * \f$f(x_1,\ldots,x_d,y)=\psi_{\alpha_{d+1}}(y)\prod_{j=1}^d\psi_{\alpha_j}(x_j)\f$,
 * which is `Homogeneous`
 * - All offdiagonal univariate functions are identical, i.e.
 * \f$f(x_1,\ldots,x_d,y)=\psi^{diag}_{\alpha_{d+1}}(y)\prod_{j=1}^d\psi^{offdiag}_{\alpha_j}(x_j)\f$,
 * which is `OffdiagHomogeneous`.
 * - All univariate basis functions may be different, i.e.
 * \f$f(x_1,\ldots,x_d,y)=\psi^{d+1}_{\alpha_{d+1}}(y)\prod_{j=1}^d\psi^{j}_{\alpha_j}(x_j)\f$
 * which is `Heterogeneous`
 */
enum BasisHomogeneity { Homogeneous, OffdiagHomogeneous, Heterogeneous };

/**
 * @brief Defines the identity function \f$g(x) = x\f$.
 */
class Identity{
public:

    KOKKOS_INLINE_FUNCTION static double Evaluate(double x){
        //stable implementation of std::log(1.0 + std::exp(x)) for large values
        return x;
    }

    KOKKOS_INLINE_FUNCTION static double Derivative(double x){
        return 1.0;
    }

    KOKKOS_INLINE_FUNCTION static double SecondDerivative(double x){
        return 0.;
    }

    KOKKOS_INLINE_FUNCTION static double Inverse(double x){
        return x;
    }

};

/**
 * @brief Class to represent all elements of a multivariate function basis
 *
 * See BasisHomogeneity for information on options for \c HowHomogeneous .
 * The form of template parameter \c BasisEvaluatorType will depend on \c
 * HowHomogeneous See the documentation of each implementation for details on
 * what's necessary. We give the option to "rectify" the function if it depends on y (i.e. make the part dependent on x positive)
 *
 * Any univariate basis function used here must have the following functions:
 *
 * - \c EvaluateAll
 * - \c EvaluateDerivatives
 * - \c EvaluateSecondDerivatives
 *
 * See @ref OrthogonalPolynomial as an example that implements the required
 * functions
 *
 * @tparam HowHomogeneous What level of homogeneity the basis has (see
 * BasisHomogeneity for more info)
 * @tparam BasisEvaluatorType The type we need to evaluate when evaluating the
 * basis
 * @tparam RectifierType The rectification operator for diag/offdiag cross-terms
 */
template <BasisHomogeneity HowHomogeneous, typename BasisEvaluatorType, typename RectifierType=Identity>
class BasisEvaluator {
  public:
  /**
   * @brief Construct a new Basis Evaluator object
   *
   * @param dim input dimension of the multivariate basis
   * @param basis1d object(s) used to evaluate the basis
   */
  BasisEvaluator(unsigned int dim, BasisEvaluatorType basis1d) {
    // This class should not be constructed
    assert(false);
  }
  /**
   * @brief Evaluate the functions for the multivariate basis
   *
   * @param dim Which input dimension to evaluate
   * @param output_eval Memory to store output (should be size maxOrder + 1)
   * @param maxOrder Maximum basis order to evaluate
   * @param point Input point to evaluate the \c dim th basis functions at
   */
  KOKKOS_INLINE_FUNCTION void EvaluateAll(unsigned int dim, double *output_eval,
                                          int maxOrder, double point) const {
    assert(false);
  }

  /**
   * @brief Evaluate the functions for the multivariate basis
   *
   * @param dim Which input dimension to evaluate
   * @param output_eval Memory to store eval output (size maxOrder + 1)
   * @param output_diff Memory to store 1st deriv output (size maxOrder + 1)
   * @param maxOrder Maximum basis order to evaluate
   * @param point Input point to evaluate the \c dim th basis functions at
   */
  KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(unsigned int dim, double *output_eval,
                                                  double *output_diff,
                                                  int maxOrder,
                                                  double point) const {
    assert(false);
  }
  // EvaluateSecondDerivatives(dim, output_eval, output_deriv, max_order, input)

  /**
   * @brief Evaluate the functions for the multivariate basis
   *
   * @param dim Which input dimension to evaluate
   * @param output_eval Memory to store eval output (size maxOrder + 1)
   * @param output_diff Memory to store 1st deriv output (size maxOrder + 1)
   * @param output_diff_2 Memory to store 2nd deriv output (size maxOrder + 1)
   * @param maxOrder Maximum basis order to evaluate
   * @param point Input point to evaluate the \c dim th basis functions at
   */
  KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(
      unsigned int dim, double *output_eval, double *output_diff, double *output_diff_2,
      int maxOrder, double point) const {
    assert(false);
  }
#if defined(MPART_HAS_CEREAL)
  /**
   * @brief Create ability to archive a Basis Evaluator object
   *
   * @tparam Archive
   * @param ar
   */
  template <typename Archive>
  void serialize(Archive &ar) {
    assert(false);
  }
#endif
};

template<typename T>
struct GetRectifier{};

template<BasisHomogeneity HowHomogeneous, typename BasisEvaluatorType, typename RectifierType>
struct GetRectifier<BasisEvaluator<HowHomogeneous, BasisEvaluatorType, RectifierType>>{
    using type = RectifierType;
};

/**
 * @brief Basis evaluator when all univariate basis fcns are identical
 *
 * \sa OrthogonalPolynomial is an example of a valid BasisEvaluatorType
 *
 * @tparam BasisEvaluatorType Univariate type to evaluate
 */
template <typename BasisEvaluatorType>
class BasisEvaluator<BasisHomogeneity::Homogeneous, BasisEvaluatorType> {
  public:
  BasisEvaluator(unsigned int, BasisEvaluatorType const &basis1d = BasisEvaluatorType()) : basis1d_(basis1d) {}
  /**
   * @brief Helper function to construct a new Basis Evaluator object
   *
   * @tparam Args
   * @param args Arguments to construct object of type \c BasisEvaluatorType
   */
  template <typename... Args>
  BasisEvaluator(Args... args) : basis1d_(args...) {}

  // EvaluateAll(dim, output, max_order, input)
  KOKKOS_INLINE_FUNCTION void EvaluateAll(int, double *output, int max_order,
                                          double input) const {
    basis1d_.EvaluateAll(output, max_order, input);
  }

  // EvaluateDerivatives(dim, output_eval, output_deriv, max_order, input)
  KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(int, double *output,
                                                  double *output_diff,
                                                  int max_order,
                                                  double input) const {
    basis1d_.EvaluateDerivatives(output, output_diff, max_order, input);
  }
  // EvaluateSecondDerivatives(dim, output_eval, output_diff1,
  //                           output_diff2, max_order, input)
  KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(int, double *output,
                                                        double *output_diff,
                                                        double *output_diff2,
                                                        int max_order,
                                                        double input) const {
    basis1d_.EvaluateSecondDerivatives(output, output_diff, output_diff2,
                                       max_order, input);
  }

#if defined(MPART_HAS_CEREAL)
  template <typename Archive>
  void serialize(Archive &ar) {
    ar(basis1d_);
  }
#endif
  /**
   * @brief Object to evaluate 1d basis fcns
   *
   */
  BasisEvaluatorType basis1d_;
};

/**
 * @brief Basis Evaluator to evaluate diagonal and off-diagonal types different
 *
 * @tparam OffdiagEvaluatorType Type to eval offdiagonal univariate basis
 * @tparam DiagEvaluatorType Type to eval diagonal univariate basis
 * @tparam RectifyType Whether the basis is rectified or not (i.e. positivize the non-diagonal part of cross terms)
 */
template <typename OffdiagEvaluatorType, typename DiagEvaluatorType, typename Rectifier>
class BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous,
                     Kokkos::pair<OffdiagEvaluatorType, DiagEvaluatorType>,
                     Rectifier> {
  public:
  BasisEvaluator(
      unsigned int dim,
      Kokkos::pair<OffdiagEvaluatorType, DiagEvaluatorType> const &basis1d)
      : offdiag_(basis1d.first), diag_(basis1d.second), dim_(dim) {}

  /**
   * @brief Construct new Basis Evaluator object from univariate basis objects
   *
   * @param dim Number of input dimensions for evaluator
   * @param offdiag Evaluator for offdiagonal input elements
   * @param diag Evaluator for diagonal input element
   */
  BasisEvaluator(unsigned int dim, OffdiagEvaluatorType const &offdiag = OffdiagEvaluatorType(),
                 DiagEvaluatorType const &diag = DiagEvaluatorType())
      : offdiag_(offdiag), diag_(diag), dim_(dim) {}

  // EvaluateAll(dim, output, max_order, input)
  KOKKOS_INLINE_FUNCTION void EvaluateAll(unsigned int dim, double *output,
                                          int max_order, double input) const {
    if (dim < dim_ - 1)
      offdiag_.EvaluateAll(output, max_order, input);
    else
      diag_.EvaluateAll(output, max_order, input);
  }

  // EvaluateDerivatives(dim, output_eval, output_deriv, max_order, input)
  KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(unsigned int dim, double *output,
                                                  double *output_diff,
                                                  int max_order,
                                                  double input) const {
    if (dim < dim_ - 1)
      offdiag_.EvaluateDerivatives(output, output_diff, max_order, input);
    else
      diag_.EvaluateDerivatives(output, output_diff, max_order, input);
  }

  // EvaluateSecondDerivatives(dim, output_eval, output_deriv, max_order, input)
  KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(unsigned int dim, double *output,
                                                        double *output_diff,
                                                        double *output_diff2,
                                                        int max_order,
                                                        double input) const {
    if (dim < dim_ - 1)
      offdiag_.EvaluateSecondDerivatives(output, output_diff, output_diff2,
                                         max_order, input);
    else
      diag_.EvaluateSecondDerivatives(output, output_diff, output_diff2,
                                      max_order, input);
  }

#if defined(MPART_HAS_CEREAL)
  template <typename Archive>
  void serialize(Archive &ar) {
    ar(dim_, offdiag_, diag_);
  }
#endif
  /// @brief Number of input dimensions for multivariate basis
  unsigned int dim_;
  OffdiagEvaluatorType offdiag_;
  DiagEvaluatorType diag_;
};

/**
 * @brief Basis Evaluator to eval different basis fcns for arbitrary inputs
 *
 * @tparam CommonBasisEvaluatorType Supertype to univariate basis eval types
 */
template <typename CommonBasisEvaluatorType, typename Rectifier>
class BasisEvaluator<BasisHomogeneity::Heterogeneous,
                     std::vector<std::shared_ptr<CommonBasisEvaluatorType>>, Rectifier> {
  public:
  BasisEvaluator(
      unsigned int dim,
      std::vector<std::shared_ptr<CommonBasisEvaluatorType>> const &basis1d)
      : basis1d_(basis1d) {
    assert(dim == basis1d.size());  // TODO: Fix
  }
  // EvaluateAll(dim, output, max_order, input)
  // dim is zero-based indexing
  KOKKOS_INLINE_FUNCTION void EvaluateAll(unsigned int dim, double *output,
                                          int max_order, double input) const {
    basis1d_[dim]->EvaluateAll(output, max_order, input);
  }
  // EvaluateDerivatives(dim, output_eval, output_deriv, max_order, input)
  KOKKOS_INLINE_FUNCTION void EvaluateDerivatives(unsigned int dim, double *output,
                                                  double *output_diff,
                                                  int max_order,
                                                  double input) const {
    basis1d_[dim]->EvaluateDerivatives(output, output_diff, max_order, input);
  }
  // EvaluateSecondDerivatives(dim, output_eval, output_deriv, max_order, input)
  KOKKOS_INLINE_FUNCTION void EvaluateSecondDerivatives(unsigned int dim, double *output,
                                                        double *output_diff,
                                                        double *output_diff2,
                                                        int max_order,
                                                        double input) const {
    basis1d_[dim]->EvaluateSecondDerivatives(output, output_diff, output_diff2,
                                             max_order, input);
  }
#if defined(MPART_HAS_CEREAL)
  template <typename Archive>
  void serialize(Archive &ar) {
    ar(basis1d_);
  }
#endif
  // NOTE: shared_ptr is necessary to avoid type slicing
  std::vector<std::shared_ptr<CommonBasisEvaluatorType>> basis1d_;
};
}  // namespace mpart