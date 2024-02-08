#ifndef MPART_MULTIINDEXLIMITER_H_
#define MPART_MULTIINDEXLIMITER_H_

#include "MParT/MultiIndices/MultiIndex.h"

#include <functional>
#include <vector>

namespace mpart{
namespace MultiIndexLimiter{


    /** @class TotalOrderLimiter
        @brief Provides a cap on the total-order allowed
        @details This limter only allows terms that satisfy \f$\|\mathbf{j}\|_1\leq p_U\f$, where \f$\mathbf{j}\f$ is the multiindex, and \f$p_U\f$ is a nonnegative integer passed to the constructor of this class.
    */
    class TotalOrder{

    public:

        TotalOrder(unsigned int totalOrderIn) : totalOrder(totalOrderIn){};

        bool operator()(MultiIndex const& multi){return (multi.Sum() <= totalOrder);};

    private:
        const unsigned int totalOrder;

    };

    /** @class SeparableTotalOrderLimiter
        @brief Same as TotalOrder without cross-terms
        @details This limter only allows terms that satisfy
        \f$\|\mathbf{j}\|_1\leq p_U\f$, where \f$\mathbf{j}\f$
        is the multiindex, and \f$p_U\f$ is a nonnegative integer passed to the
        constructor of this class.
    */
    class SeparableTotalOrder{

    public:

        SeparableTotalOrder(unsigned int totalOrderIn) : totalOrder(totalOrderIn){};

        bool operator()(MultiIndex const& multi){
          unsigned int sum = multi.Sum();
          return (sum <= totalOrder) && (!multi.HasNonzeroEnd() || sum == multi.Get(multi.Length()-1));
        };

    private:
        const unsigned int totalOrder;

    };

    /** @class NonzeroDiagTotalOrder
        @brief Same as TotalOrder, except without any term that has nonzero diagonal entries
        @details This limter only allows terms that satisfy
        \f$\|\mathbf{j}\|_1\leq p_U\f$, where \f$\mathbf{j}\f$
        is the multiindex, and \f$p_U\f$ is a nonnegative integer passed to the
        constructor of this class.
    */
    class NonzeroDiagTotalOrder{

    public:

        NonzeroDiagTotalOrder(unsigned int totalOrderIn) : totalOrder(totalOrderIn){};

        bool operator()(MultiIndex const& multi){
          unsigned int sum = multi.Sum();
          return (sum <= totalOrder) && (multi.HasNonzeroEnd());
        };

    private:
        const unsigned int totalOrder;

    };


    /** @class Dimension
    @ingroup MultiIndices
    @brief Provides bounds on what dimensions are allowed to have nonzero values.
    @details This limiter only allows terms that satisfy \f$\mathbf{j}_d = 0 \f$ for \f$d<D_L\f$ or \f$d>=D_L+M\f$ for a lower bound \f$D_L\f$ and length \f$M\f$.
    */
    class Dimension {

    public:

        Dimension(unsigned int lowerDimIn, unsigned int lengthIn) : lowerDim(lowerDimIn), length(lengthIn){};

        bool operator()(MultiIndex const& multi) const;

    private:

        const unsigned int lowerDim;
        const unsigned int length;

    };


    /** @class Anisotropic
    @brief Declares multiindices as feasible if their entries for less important dimensions are not too high.
    @details Given a weight vector \f$ w = (w_i)_{i=1}^d \f$ with \f$ w_i \in [0,1] \f$ and a cutoff threshold \f$ \epsilon \in (0,1)\f$,
                this limiter declares a multiindex \f$ \nu = (\nu_i)_{i=1}^d \f$ as feasible if
                \f$ w^\nu := \prod_{i=1}^d w_i^{\nu_i} > \epsilon \f$.
                It thus implements the multiindex selection criterion for the construction of a priori anisotropic sparse grids as
                described in Algorithm 2 in <ul><li> Zech, Jakob. <i>Sparse-grid approximation of high-dimensional parametric PDEs.</i> ETH Zurich, 2018. </li></ul>
    */
    class Anisotropic{

    public:

        Anisotropic(std::vector<double> const& weightsIn,
                    double                     epsilonIn);

        bool operator()(MultiIndex const& multi) const;

    private:
        const std::vector<double> weights;
        const double epsilon;

    };


    /** @class MaxDegree
    @ingroup MultiIndices
    @brief Provides a cap on the maximum value of each component the multiindex
    @details This limter only allows terms that satisfy \f$\mathbf{j}_i\leq p_i\f$ for \f$i\in \{1,2,\ldots,D\}\f$, where \f$p\f$ is a vector of upper bounds.
    */
    class MaxDegree{

    public:
        MaxDegree(unsigned int maxDegreeIn, unsigned int length) : MaxDegree(std::vector<unsigned int>(length, maxDegreeIn)){};
        MaxDegree(std::vector<unsigned int> const& maxDegreesIn) : maxDegrees(maxDegreesIn){};

        bool operator()(MultiIndex const& multi) const;

    private:
        std::vector<unsigned int> maxDegrees;

    };

  /** @class None
   @ingroup MultiIndices
   @brief Returns true for any multiindex
   @details This class is used as a default in many places where a limiter is not always needed.  IsFeasible will return true for any multiindex.
   */
 class None{
  public:
    bool operator()(MultiIndex const&) const {return true;};
  };

  /** @class And
   @ingroup MultiIndices
   @brief Combines two limiters through an AND operation
   @details This class will return true if both limiters given to the constructor return true.
   */
 class And{

  public:
    And(std::function<bool(MultiIndex const&)> limitA,
        std::function<bool(MultiIndex const&)> limitB) : a(limitA), b(limitB){};

    bool operator()(MultiIndex const& multi) const {return (a(multi) && b(multi));};

  private:
    std::function<bool(MultiIndex const&)> a, b;

  };


  /** @class Or
   @ingroup MultiIndices
   @brief Combines two limiters through an OR operation
   @details This class will return true if either of the limiters given to the constructor return true.
   */
  class Or{

   public:
     Or(std::function<bool(MultiIndex const&)> limitA,
        std::function<bool(MultiIndex const&)> limitB) : a(limitA), b(limitB){};

    bool operator()(MultiIndex const& multi) const {return (a(multi) || b(multi));};

  private:
    std::function<bool(MultiIndex const&)> a, b;

  };


  /** @class Xor
   @ingroup MultiIndices
   @brief Combines two limiters through an XOR operation
   @details This class will return true if exactly one of the limiters given to the constructor returns true.
   */
  class Xor{

  public:
  public:
    Xor(std::function<bool(MultiIndex const&)> limitA,
        std::function<bool(MultiIndex const&)> limitB) : a(limitA), b(limitB){};

    bool operator()(MultiIndex const& multi) const {return (a(multi) ^ b(multi));};

  private:
    std::function<bool(MultiIndex const&)> a, b;

  };

} // namespace MultiIndexLimiter
} // namespace mpart

#endif
