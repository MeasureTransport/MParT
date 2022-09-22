#ifndef MPART_IDENTITYMAP_H
#define MPART_IDENTITYMAP_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/Miscellaneous.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>


namespace mpart{

/**
 @brief Provides a definition of the identity map.
 @details 
This class defines the identity map \f$I:\mathbb{R}^N\rightarrow \mathbb{R}^M\f$, i.e., a map such that \f$I(x_{1:N-M},x_{N-M:N}) = x_{N-M:N}\f$ 


 */
template<typename MemorySpace>
class IdentityMap : public ConditionalMapBase<MemorySpace>
{
public:

    /** @brief Construct an map that acts as the identify.

         @param inDim The dimension \f$N\f$ of the input to this map.
         @param outDim The dimension \f$M\f$ of the output from this map.
    */
    IdentityMap(unsigned int inDim, unsigned int outDim);


    void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                      StridedMatrix<double, MemorySpace>              output);

    void InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                     StridedMatrix<const double, MemorySpace> const& r,
                     StridedMatrix<double, MemorySpace>              output);


    void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                            StridedVector<double, MemorySpace>              output);

    void CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                       StridedMatrix<const double, MemorySpace> const& sens,
                       StridedMatrix<double, MemorySpace>              output);
    
    
    void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                       StridedMatrix<const double, MemorySpace> const& sens,
                       StridedMatrix<double, MemorySpace>              output);


    void LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                     StridedMatrix<double, MemorySpace>              output);

    void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                     StridedMatrix<double, MemorySpace>              output);

}; // class IdentityMap

}

#endif