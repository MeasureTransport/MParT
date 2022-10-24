#include "MParT/SingleEntryMap.h"
#include "MParT/TriangularMap.h"
#include "MParT/IdentityMap.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
SingleEntryMap<MemorySpace>::SingleEntryMap(const unsigned int dim, const unsigned int activeInd, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& component) : 
                        ConditionalMapBase<MemorySpace>(dim, dim, component->numCoeffs)
{

    // Check that active index is not greater than map dimension
    if(dim < activeInd){
        std::stringstream msg;
        msg << "In SingleEntryMap constructor, the active index can't be greater than map dimension. Got dim = " << dim << " and activeInd = " << activeInd << ".";
        throw std::invalid_argument(msg.str());
    }

    // Check that the input dimension of the component matches the activeInd
    if(activeInd != component->inputDim){
        std::stringstream msg;
        msg << "In SingleEntryMap constructor, the component input dimension must be equal to the active index. Got dim = " << component->inputDim << " and activeInd = " << activeInd << ".";
        throw std::invalid_argument(msg.str());
    }

    std::shared_ptr<ConditionalMapBase<MemorySpace>> map;
    // Construct map using SingleEntryMap constructor
    
    if(activeInd == 1){  // special case if activeInd = 1, map is of form [T_1; Id]

        // Bottom identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> botIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(dim, dim-activeInd);
        
        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(2);
        blocks.at(0) = component;
        blocks.at(1) = botIdMap;

        // make map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = std::make_shared<TriangularMap<MemorySpace>>(blocks);


    }
    else if (activeInd == dim){  // special case if activeInd = dim, map is of form [Id; T_d]
        // Top identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> topIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(activeInd-1, activeInd-1);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(2);
        blocks.at(0) = topIdMap;
        blocks.at(1) = component;

        // make map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = std::make_shared<TriangularMap<MemorySpace>>(blocks);
    }
    else{ // general case, map is of form [Id; T_i; Id]

        // Top identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> topIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(activeInd-1, activeInd-1);
        
        // Bottom identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> botIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(dim, dim-activeInd);
        
        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(3);
        blocks.at(0) = topIdMap;
        blocks.at(1) = component;
        blocks.at(2) = botIdMap;

        // make map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> map = std::make_shared<TriangularMap<MemorySpace>>(blocks);


    }
    
    map_(map);
}



template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    map_->SetCoeffs(coeffs);
}

template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    map_->WrapCoeffs(coeffs);
}

#if defined(MPART_ENABLE_GPU)
template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{
    map_->SetCoeffs(coeffs);
}

template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{
    map_->WrapCoeffs(coeffs);
}
#endif

template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                    StridedVector<double, MemorySpace>              output)
{   
    map_->LogDeterminantImpl(pts, output);
}

template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{
    map_->EvaluateImpl(pts, output);
}

template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                             StridedMatrix<const double, MemorySpace> const& r,
                                             StridedMatrix<double, MemorySpace>              output)
{

    map_->InverseImpl(x1, r, output);
}



template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<const double, MemorySpace> const& sens,
                                              StridedMatrix<double, MemorySpace>              output)
{
    map_->GradientImpl(pts, sens, output);
}

template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{
    map_->CoeffGradImpl(pts, sens, output);
}

template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{
    map_->LogDeterminantCoeffGradImpl(pts, output);
}

template<typename MemorySpace>
void SingleEntryMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{
    map_->LogDeterminantInputGradImpl(pts, output);
}

// Explicit template instantiation
template class mpart::SingleEntryMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::SingleEntryMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif

