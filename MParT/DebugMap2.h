#ifndef MPART_DEBUGMAP_H
#define MPART_DEBUGMAP_H

#include "MParT/ConditionalMapBase.h"
#include "MParT/Utilities/Miscellaneous.h"

#include <Eigen/Core>

#include <Kokkos_Core.hpp>


namespace mpart{

/**
 @brief Provides a definition of block lower triangular transport maps.
 @details
This class defines a map \f$T:\mathbb{R}^N\rightarrow \mathbb{R}^M\f$ with the block triangular structure
\f[
T(x) = \left[\begin{array}{l} T_1(x_{1:N_1})\\ \vdots \\ T_k(x_{1:N_2})\\ \vdots T_K(x_{1:N}) \end{array} \right],
\f]
where each component \f$T_i(x_{1:N_i}):\mathbb{R}^{N_i}\rightarrow \mathbb{R}^{M_i}\f$ is a function depending on
the first \f$N_i\f$ inputs and returning \f$M_i\f$ outputs.  Note that this function must be invertible in the last
\f$M_i\f$ input arguments.  For example, it must be possible to solve \f$T_i(x_1:{N_i-M_i}, x_{N_i-M_i:N_i}) = r\f$
for \f$x_{N_i-M_i:N_i}\f$ given a vector \f$r\in\mathbb{R}^{M_i}\f$ and previous components \f$x_1:{N_i-M_i}\f$.

This block triangular form is analogous to a block triangular matrix where each \f$M_i\timesM_i\f$ diagonal block
is positive definite.

 */
template<typename MemorySpace>
class DebugMap : public ConditionalMapBase<MemorySpace>{
        
public:

    /** @brief Construct a block triangular map from a collection of other ConditionalMapBase objects.

    @param components A vector of ConditionalMapBase objects defining each \f$T_k\f$ in the block triangular map.
                      To maintain the correct block structure, the dimensions of the components must satisfy \f$N_k = N_{k-1}+M_{k}\f$.
    */
    DebugMap(std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const& summary, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& component);

    virtual ~DebugMap() = default;

    /** @brief Sets the coefficients for all components of the map.

    @details This function will copy the provided coeffs vectors into the savedCoeffs object in the DebugMap class.   To avoid
    duplicating the coefficients, the savedCoeffs member variable for each component will then be set to a subview of this vector.
    @param coeffs A vector containing coefficients for all components.  If component \f$k\f$ is defined by \f$C_k\f$ coefficients,
                  then this vector should have length \f$\sum_{k=1}^K C_i\f$ and the coefficients for component \f$k\f$ should
                  start at index \f$\sum_{j=1}^{k-1} C_j\f$.
    */
    using ConditionalMapBase<MemorySpace>::SetCoeffs;
    virtual void SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs) override;
    virtual void WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs) override;
    #if defined(MPART_ENABLE_GPU)
    virtual void SetCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs) override;
    virtual void WrapCoeffs(Kokkos::View<double*, mpart::DeviceSpace> coeffs) override;
    #endif 
    


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


    void print_ptr() { std::cout << "print_ptr(): comp_.get() = " << comp_.get() << std::endl; std::cout << "print_ptr(): summary_.get() = " << summary_.get() << std::endl; }
private:
 
    std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> summary_;
    std::shared_ptr<ConditionalMapBase<MemorySpace>> comp_;
    


}; // class DebugMap

}

#endif