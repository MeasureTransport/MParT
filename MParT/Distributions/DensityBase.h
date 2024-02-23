#ifndef MPART_DensityBase_H
#define MPART_DensityBase_H

#include "MParT/Utilities/EigenTypes.h"
#include "MParT/Utilities/ArrayConversions.h"

#include "MParT/Utilities/GPUtils.h"

#include <Eigen/Core>

namespace mpart {

/**
 * @brief A base class to represent a density, a function \f$p:\mathbb{R}^m\to\mathbb{R}\f$, which usually (but
 *        does not have to) integrates to unity over the input space. Any density must contain a LogDensityImpl
 *        and LogDensityInputGradImpl function.
 *
 * @tparam MemorySpace Where the density stores data for computation.
 */
template <typename MemorySpace>
class DensityBase {

public:

    /**
     * @brief Construct a new Density Base object with dimension dim
     *
     * @param dim the dimension of the input to the density
     */
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
    StridedVector<double, MemorySpace> LogDensity(StridedMatrix<const double, MemorySpace> const &X);
    #if defined(MPART_ENABLE_GPU)
    StridedVector<double, std::conditional_t< std::is_same_v<MemorySpace, Kokkos::HostSpace> , DeviceSpace, Kokkos::HostSpace> > LogDensity(StridedMatrix<const double, std::conditional_t< std::is_same_v<MemorySpace, Kokkos::HostSpace> , DeviceSpace, Kokkos::HostSpace > > const &X);
    #endif

    /**
     * @brief Computes the log density at the given points.
     * @param pts The points where we want to evaluate the gradient of the log density.
     * @param output The matrix where we want to store the gradient of the log density.
     */
    virtual void LogDensityInputGradImpl(StridedMatrix<const double, MemorySpace> const &pts, StridedMatrix<double, MemorySpace> output) = 0;

    /** LogDensity function with conversion from Eigen to Kokkos (and possibly copy to/from device). */
    Eigen::RowMatrixXd LogDensityInputGrad(Eigen::Ref<const Eigen::RowMatrixXd> const &pts);

    /**
     * @brief Computes the gradient of the log density at the given points.
     * @details For this density \f$p:\mathbb{R}^m\to\mathbb{R}\f$, computes \f$\nabla_x\log p(x)\f$ at each column of \f$X\in\mathbb{R}^{m\times n}\f$, where \f$n\f$ is the number of points we want to evaluate the density gradient at.
     * @param X The points where we want to evaluate the gradient log density.
     * @return A matrix \f$A\in\mathbb{R}^{m\times n}\f$ containing the gradient of the log density at each point.
     */
    StridedMatrix<double, MemorySpace> LogDensityInputGrad(StridedMatrix<const double, MemorySpace> const &X);
    #if defined(MPART_ENABLE_GPU)
    StridedMatrix<double, std::conditional_t< std::is_same_v<MemorySpace, Kokkos::HostSpace> , DeviceSpace, Kokkos::HostSpace> > LogDensityInputGrad(StridedMatrix<const double, std::conditional_t< std::is_same_v<MemorySpace, Kokkos::HostSpace> , DeviceSpace, Kokkos::HostSpace > > const &X);
    #endif

    /**
     * @brief Returns the input dimension of the density
     *
     * @return unsigned int dimension of the density
     */
    virtual unsigned int Dim() const { return dim_; }

protected:
    const unsigned int dim_;
};
}

#endif // MPART_DensityBase_H
