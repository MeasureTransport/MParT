#ifndef MPART_AFFINEFUNCTION_H
#define MPART_AFFINEFUNCTION_H

#include <Kokkos_Core.hpp>

#include "MParT/ParameterizedFunctionBase.h"

#include <Eigen/Core>
#include <Eigen/Dense>


namespace mpart{

/** @brief Defines functions of the form \f$Ax+b\f$ for an arbitrary rectangular matrix \f$A\f$ and vector offset \f$b\f$.
*/
template<typename MemorySpace>
class AffineFunction : public ParameterizedFunctionBase<MemorySpace>
{
public:
    /** Construct a function that simply shifts the input.
    \f[ y = x + b \f]
    */
    AffineFunction(StridedVector<double,MemorySpace> b);

    /** Constructs a linear function that only scales the input.
    \f[ y = Ax\f]
    */
    AffineFunction(StridedMatrix<double,MemorySpace> A);

    /** Constructs an affine function that scales and shifts the input.
    \f[ y = Ax + b \f]
    */
    AffineFunction(StridedMatrix<double,MemorySpace> A, StridedVector<double,MemorySpace> b);

    virtual ~AffineFunction() = default;

    void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<double, MemorySpace>              output) override;

    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                       StridedMatrix<const double, MemorySpace> const& sens,
                       StridedMatrix<double, MemorySpace>              output) override;

    void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<const double, MemorySpace> const& sens,
                      StridedMatrix<double, MemorySpace>              output) override;




protected:

    StridedMatrix<double,MemorySpace> A_;
    StridedVector<double,MemorySpace> b_;

    int ldA;

};

}
#endif