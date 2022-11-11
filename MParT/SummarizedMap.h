#ifndef MPART_SUMMARIZEDMAP_H
#define MPART_SUMMARIZEDMAP_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/Miscellaneous.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>


namespace mpart{

/**
 @brief Provides a definition for a map with 'summary structure'.
 @details
This class defines a map \f$T:\mathbb{R}^N\rightarrow \mathbb{R} \f$ with the summary structure
\f[
T(x) = \tilde{T}(s(x_{< N}), x_N)
\f]
where the function \f$s:\mathbb{R}^{N-1}\rightarrow \mathbb{R}^{r}\f$ is a function that summarizes the leading \f$N-1\f$ inputs (\f$ x_{< N} \f$).


 */
template<typename MemorySpace>
class SummarizedMap : public ConditionalMapBase<MemorySpace>{

public:

    /** @brief Constructs a map with 'summary structure'.

    @details Constructs a map \f$T:\mathbb{R}^N\rightarrow \mathbb{R} \f$ with the summary structure
\f[
T(x) = \tilde{T}(s(x_{< N}), x_N)
\f]
where the function \f$s:\mathbb{R}^{N-1}\rightarrow \mathbb{R}^{r}\f$ is a function that summarizes the leading \f$N-1\f$ inputs (\f$ x_{< N} \f$).

    @param summary A ParameterizedFunctionBase object that defines the summary function \f$ s \f$.
    @param component ConditionalMapBase object defining the component \f$\tilde{T}\f$.

    */
    SummarizedMap(std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const& summary, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& component);

    virtual ~SummarizedMap() = default;

    using ConditionalMapBase<MemorySpace>::SetCoeffs;
    virtual void SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs) override;
    virtual void WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs) override;
    #if defined(MPART_ENABLE_GPU)
    virtual void SetCoeffs(Kokkos::View<double*, DeviceSpace> coeffs) override;
    virtual void WrapCoeffs(Kokkos::View<double*, DeviceSpace> coeffs) override;
    #endif

    void SummarizePts(StridedMatrix<const double, MemorySpace> const& pts,
                                    StridedMatrix<double, MemorySpace>              output);


    virtual void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                    StridedVector<double, MemorySpace>              output) override;


    virtual void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<double, MemorySpace>              output) override;

    virtual void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                              StridedMatrix<const double, MemorySpace> const& sens,
                              StridedMatrix<double, MemorySpace>              output) override;


    virtual void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                             StridedMatrix<const double, MemorySpace> const& r,
                             StridedMatrix<double, MemorySpace>              output) override;


    virtual void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                               StridedMatrix<const double, MemorySpace> const& sens,
                               StridedMatrix<double, MemorySpace>              output) override;


    virtual void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                             StridedMatrix<double, MemorySpace>              output) override;

    virtual void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                             StridedMatrix<double, MemorySpace>              output) override;


private:

    std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const sumFunc_;
    std::shared_ptr<ConditionalMapBase<MemorySpace>> const comp_;



}; // class SummarizedMap

}

#endif