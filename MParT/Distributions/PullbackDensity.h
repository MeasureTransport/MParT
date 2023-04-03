#ifndef MPART_PULLBACKDENSITY_H
#define MPART_PULLBACKDENSITY_H

#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/DensityBase.h"
#include "MParT/Utilities/LinearAlgebra.h"
#include "MParT/ConditionalMapBase.h"

namespace mpart {

/**
 * @brief A class to represent a density of a Pullback distribution given a map and a density.
 *        If \f$X\sim\mu\f$, then \f$T\f$ satisfies \f$T(X)\sim\nu\f$ and this class represents the density of the distribution
 *        \f$\mu\f$ given the density of \f$\nu\f$ and the map \f$T\f$.
 *
 * @tparam MemorySpace Where data is stored for computation
 */
template<typename MemorySpace>
class PullbackDensity: public DensityBase<MemorySpace> {

    public:
    PullbackDensity() = delete;
    /**
     * @brief Construct a new Pullback Density object given the transport map and the density \f$\nu\f$
     *
     * @param map transport map \f$T:\mu\to\nu\f$
     * @param density density of the distribution \f$\nu\f$
     */
    PullbackDensity(std::shared_ptr<ConditionalMapBase<MemorySpace>> map, std::shared_ptr<DensityBase<MemorySpace>> density);

    /**
     * @brief Given points distributed according to \f$\mu\f$, calculate the log PDF of those points using the pullback
     *
     * @param pts (MxN) data matrix where each column is identically distributed according to \f$\mu\f$
     * @param output N-length vector to store the log density evaluation
     */
    void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) override;

    /**
     * @brief Given points distributed according to \f$\mu\f$, calculate the gradient of the log PDF of those points using the pullback wrt the input
     *
     * @param pts (MxN) data matrix where each column is identically distributed according to \f$\mu\f$
     * @param output (MxN) matrix where entry (i,j) is the derivative of the density of \f$\nu\f$ with respect to the i-th input evaluated at the j-th sample
     */
    void LogDensityInputGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) override;

    /**
     * @brief The derivative of the pullback distribution density with respect to the parameters (i.e. coefficients) of the map
     *
     * @param pts data matrix where each column is identically distributed according to \f$\mu\f$
     * @param output memory to place the derivative of the pullback action on pts w.r.t. the parameters of the map
     */
    void LogDensityCoeffGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output);

    /**
     * @brief The derivative of the pullback distribution density with respect to the parameters (i.e. coefficients) of the map
     *
     * @param pts data matrix where each column is identically distributed according to \f$\mu\f$
     * @return StridedMatrix<double, MemorySpace> derivative of the pullback action on pts w.r.t. the parameters of the map
     */
    StridedMatrix<double, MemorySpace> LogDensityCoeffGrad(StridedMatrix<const double, MemorySpace> const &pts);

    /**
     * @brief The derivative of the pullback distribution density with respect to the parameters (i.e. coefficients) of the map
     *
     * @param pts data matrix where each column is identically distributed according to \f$\mu\f$
     * @return Eigen::RowMatrixXd derivative of the pullback action on pts w.r.t. the parameters of the map
     */
    Eigen::RowMatrixXd LogDensityCoeffGrad(Eigen::Ref<const Eigen::RowMatrixXd> const &pts);

    private:
    /**
     * @brief The map T that pushes \f$\mu\f$ to \f$\nu\f$.
     *
     */
    std::shared_ptr<ConditionalMapBase<MemorySpace>> map_;

    /**
     * @brief The distribution \f$\nu\f$
     *
     */
    std::shared_ptr<DensityBase<MemorySpace>> density_;
};

} // namespace mpart

#endif //MPART_PULLBACKDENSITY_H