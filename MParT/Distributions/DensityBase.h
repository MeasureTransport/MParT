#ifndef MPART_DensityBase_H
#define MPART_DensityBase_H

#include "MParT/Utilities/EigenTypes.h"
#include "MParT/Utilities/ArrayConversions.h"

#include "MParT/Utilities/GPUtils.h"

#include <Eigen/Core>

namespace mpart {

    template <typename MemorySpace>
    class DensityBase {

    public:

    DensityBase(unsigned int dim): dim_(dim) {};

    virtual ~DensityBase() = default;

    /**
     * @brief Computes the log density at the given points.
     * @param pts The points where we want to evaluate the log density.
     * @param output The output log density values.
     */
    virtual void LogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedVector<double, MemorySpace> output) = 0;

    /** LogDensity function with conversion from Eigen to Kokkos (and possibly copy to/from device). */
    Eigen::VectorXd LogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts);

    /**
     * @brief Computes the log density at the given points.
     * @details For this density \f$p:\mathbb{R}^m\to\mathbb{R}\f$, computes \f$\log p(x)\f$ at each column of \f$X\in\mathbb{R}^{m\times n}\f$, where \f$n\f$ is the number of points we want to evaluate the density at.
     * @param X The points where we want to evaluate the log density.
     * @return Matrix \f$A\in\mathbb{R}^{m\times n}\f$ containing the log density at each point.
     */
    template<typename AnyMemorySpace>
    StridedVector<double, AnyMemorySpace> LogDensity(StridedMatrix<const double, AnyMemorySpace> const &X);

    /**
     * @brief Computes the log density at the given points.
     * @param pts The points where we want to evaluate the gradient of the log density.
     * @param output The matrix where we want to store the gradient of the log density.
     */
    virtual void GradLogDensityImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) = 0;

    /** LogDensity function with conversion from Eigen to Kokkos (and possibly copy to/from device). */
    Eigen::RowMatrixXd GradLogDensity(Eigen::Ref<const Eigen::RowMatrixXd> const &pts);

    /**
     * @brief Computes the gradient of the log density at the given points.
     * @details For this density \f$p:\mathbb{R}^m\to\mathbb{R}\f$, computes \f$\nabla_x\log p(x)\f$ at each column of \f$X\in\mathbb{R}^{m\times n}\f$, where \f$n\f$ is the number of points we want to evaluate the density gradient at.
     * @param X The points where we want to evaluate the gradient log density.
     * @return A matrix \f$A\in\mathbb{R}^{m\times n}\f$ containing the gradient of the log density at each point.
     */
    template<typename AnyMemorySpace>
    StridedMatrix<double, AnyMemorySpace> GradLogDensity(StridedMatrix<const double, AnyMemorySpace> const &X);

    virtual unsigned int Dim() const { return dim_; }

    protected:
    const unsigned int dim_;
    };
}

#endif // MPART_DensityBase_H