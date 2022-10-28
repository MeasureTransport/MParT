#include "MParT/DebugMap.h"

#include "MParT/Utilities/KokkosSpaceMappings.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
DebugMap<MemorySpace>::DebugMap(std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const& summary, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& component) : ConditionalMapBase<MemorySpace>(component->inputDim, component->outputDim, component->numCoeffs),
                        summary_(summary), comp_(component)
{

    std::cout << "constructor: comp_.get() = " << comp_.get() << std::endl;
    std::cout << "constructor: summary_.get() = " << summary_.get() << std::endl; 
}

template<typename MemorySpace>
void DebugMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);

    comp_->SetCoeffs(coeffs);
}

template<typename MemorySpace>
void DebugMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{

    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::WrapCoeffs(coeffs);

    comp_->WrapCoeffs(coeffs);

}

#if defined(MPART_ENABLE_GPU)
template<typename MemorySpace>
void DebugMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{

        // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);

    comp_->SetCoeffs(coeffs);

}

template<typename MemorySpace>
void DebugMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::WrapCoeffs(coeffs);

    comp_->SetCoeffs(coeffs);
}
#endif

template<typename MemorySpace>
void DebugMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                    StridedVector<double, MemorySpace>              output)
{   

    comp_->LogDeterminantImpl(pts, output);

}

template<typename MemorySpace>
void DebugMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{
    comp_->EvaluateImpl(pts, output); 
}

template<typename MemorySpace>
void DebugMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                             StridedMatrix<const double, MemorySpace> const& r,
                                             StridedMatrix<double, MemorySpace>              output)
{

    comp_->InverseImpl(x1, r, output); 
}


template<typename MemorySpace>
void DebugMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<const double, MemorySpace> const& sens,
                                              StridedMatrix<double, MemorySpace>              output)
{

    comp_->GradientImpl(pts, sens, output); 

}

template<typename MemorySpace>
void DebugMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{
    comp_->CoeffGradImpl(pts, sens, output); 
}

template<typename MemorySpace>
void DebugMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{
    comp_->LogDeterminantCoeffGradImpl(pts, output); 
}

template<typename MemorySpace>
void DebugMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{
    comp_->LogDeterminantInputGradImpl(pts, output); 
}

// Explicit template instantiation
template class mpart::DebugMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::DebugMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif