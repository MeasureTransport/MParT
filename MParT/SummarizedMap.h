#ifndef MPART_SUMMARIZEDMAP_H
#define MPART_SUMMARIZEDMAP_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/Miscellaneous.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>


namespace mpart{

/**
 @brief Provides a definition of a transport map with summary structure.
 @details
This class defines a map \f$T:\mathbb{R}^N\rightarrow \mathbb{R}^M\f$ with the block triangular structure
\f[
T(x_1,x_2) = T'(s(x_1),x_2) 
\f]
where \f$s:\mathbb{R}^{N_1} \to \mathbb{R}^{r}\f$, where \f$ r< N_1 \f$.


 */
template<typename MemorySpace>
class SummarizedMap : public ConditionalMapBase<MemorySpace>{
        
public:

    /** @brief Construct a block triangular map from a collection of other ConditionalMapBase objects.

    @param summaryFunction The function \f$s\f$ defining the summary used in the map.
    */
    SummarizedMap(std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const& summaryFunction, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& map);

    virtual ~SummarizedMap() = default;


    using ConditionalMapBase<MemorySpace>::SetCoeffs;
    virtual void SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs) override;
    virtual void WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs) override;
    #if defined(MPART_ENABLE_GPU)
    virtual void SetCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs) override;
    virtual void WrapCoeffs(Kokkos::View<double*, mpart::DeviceSpace> coeffs) override;
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

    std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const& summaryFunction_;
    std::shared_ptr<ConditionalMapBase<MemorySpace>> const & map_;


}; // class SummarizedMap

}

#endif