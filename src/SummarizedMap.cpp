#include "MParT/SummarizedMap.h"

#include "MParT/Utilities/KokkosSpaceMappings.h"
#include <MParT/Utilities/ArrayConversions.h>
#include <numeric>

using namespace mpart;

template<typename MemorySpace>
SummarizedMap<MemorySpace>::SummarizedMap(std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const& summaryFunction, 
                                          std::shared_ptr<ConditionalMapBase<MemorySpace>> const& map) : 
                                          ConditionalMapBase<MemorySpace>(summaryFunction->inputDim + 1, map->outputDim, map->numCoeffs),
                                          summaryFunction_(summaryFunction), 
                                          map_(map)
{

    // Check the dimension of summaryFunction is compatible with dimension of component
    if(map_->outputDim != 1){
        std::stringstream msg;
        msg << "SummarizedMap currently supports output dimension = 1 only, but was given " << map_->outputDim << ".";
        throw std::invalid_argument(msg.str());
    }

    if(map_->inputDim != summaryFunction->outputDim + 1){
        std::stringstream msg;
        msg << "SummarizedMap: input dimension of map component must be 1 + output dimension of summaryFunction, but was given map->inputDim = " << map_->inputDim << " and summaryFunction->outputDim + 1 = " << summaryFunction_->outputDim + 1 << ".";
        throw std::invalid_argument(msg.str());
    }
}



template<typename MemorySpace>
void SummarizedMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    // SetCoeffs for the component (not the summary function)
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);
    map_->WrapCoeffs(this->savedCoeffs);

    
}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    // WrapCoeffs for the component (not the summary function)
    ConditionalMapBase<MemorySpace>::WrapCoeffs(coeffs);
    map_->WrapCoeffs(this->savedCoeffs);
}

#if defined(MPART_ENABLE_GPU)
template<typename MemorySpace>
void SummarizedMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{
    // SetCoeffs for the component (not the summary function)
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);
    map_->WrapCoeffs(this->savedCoeffs);
}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{
    // WrapCoeffs for the component (not the summary function)
    ConditionalMapBase<MemorySpace>::WrapCoeffs(coeffs);
    map_->WrapCoeffs(coeffs);
}
#endif


template<typename MemorySpace>
void SummarizedMap<MemorySpace>::SummarizePts(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{   
    // Split pts into the part that is summarized and the part that is not

    StridedMatrix<const double, MemorySpace> ptsToSummarize = Kokkos::subview(pts, std::make_pair(0, int(summaryFunction_->inputDim)), Kokkos::ALL());
    StridedMatrix<const double, MemorySpace> ptsAfterSummary = Kokkos::subview(pts, std::make_pair(int(summaryFunction_->inputDim), int(summaryFunction_->inputDim+1)), Kokkos::ALL());

    // Evaluate summary function
    Kokkos::View<double**, MemorySpace> summary = summaryFunction_->Evaluate(ptsToSummarize);

    // Copy summarized pts
    Kokkos::View<double**, MemorySpace> outputSummary = Kokkos::subview(output, std::make_pair(0,int(summaryFunction_->outputDim)), Kokkos::ALL());
    Kokkos::deep_copy(outputSummary, summary);

    // Copy non summarized pts
    Kokkos::View<double**, MemorySpace> outputAfterSummary = Kokkos::subview(output, std::make_pair(int(summaryFunction_->outputDim),int(summaryFunction_->outputDim+1)), Kokkos::ALL());        
    Kokkos::deep_copy(outputAfterSummary, ptsAfterSummary);

}


template<typename MemorySpace>
void SummarizedMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                    StridedVector<double, MemorySpace>              output)
{   

    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", map_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts,summarizedPts);

    // LogDeterminant of map
    map_->LogDeterminantImpl(summarizedPts, output);

}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{

    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", map_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts, summarizedPts);

    // Evaluate map
    map_->EvaluateImpl(summarizedPts, output);
    
}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                             StridedMatrix<const double, MemorySpace> const& r,
                                             StridedMatrix<double, MemorySpace>              output)
{

    // Create a view to hold pts to summarize
    StridedMatrix<const double, MemorySpace> ptsToSummarize = Kokkos::subview(x1, std::make_pair(0, int(summaryFunction_->inputDim)), Kokkos::ALL());

    // // Evaluate summary function 
    Kokkos::View<double**, MemorySpace> summary = summaryFunction_->Evaluate(ptsToSummarize);

    // Invert map
    map_->InverseImpl(summary, r, output);
}



template<typename MemorySpace>
void SummarizedMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<const double, MemorySpace> const& sens,
                                              StridedMatrix<double, MemorySpace>              output)
{

    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", map_->inputDim, pts.extent(1));

    // Summarize the pts 
    this->SummarizePts(pts, summarizedPts);

    // Create a view for the gradient of map_ wrt s(x_1) and x_2
    Kokkos::View<double**, MemorySpace> outputForSummaryAndX2("outputForSummaryAndX2", map_->inputDim, pts.extent(1));
    
    // GradientImpl of map
    map_->GradientImpl(summarizedPts, sens, outputForSummaryAndX2);

    // Split outputForSummaryAndX2 into summary and x2 parts
    Kokkos::View<double**, MemorySpace> outputForSummary = Kokkos::subview(outputForSummaryAndX2, std::make_pair(0,int(summaryFunction_->outputDim)), Kokkos::ALL());
    Kokkos::View<double**, MemorySpace> outputForX2 = Kokkos::subview(outputForSummaryAndX2, std::make_pair(int(summaryFunction_->outputDim),int(summaryFunction_->outputDim+1)), Kokkos::ALL());
    
    // Create a view for the gradient of map_ wrt x_1
    Kokkos::View<double**, MemorySpace> outputForX1("outputForX1", summaryFunction_->inputDim, pts.extent(1));
    
    // GradientImpl of summary using outputForSummary as the sens. vectors (from chain-rule)
    StridedMatrix<const double, MemorySpace>  x1 = Kokkos::subview(pts, std::make_pair(0,int(summaryFunction_->inputDim)), Kokkos::ALL());
    summaryFunction_->GradientImpl(x1, outputForSummary, outputForX1);

    // Copy results over to the output
    Kokkos::View<double**, MemorySpace> outputSubX1 = Kokkos::subview(output, std::make_pair(0,int(summaryFunction_->inputDim)), Kokkos::ALL());
    Kokkos::deep_copy(outputSubX1, outputForX1);

    Kokkos::View<double**, MemorySpace> outputSubX2 = Kokkos::subview(output, std::make_pair(int(summaryFunction_->inputDim),int(summaryFunction_->inputDim+1)), Kokkos::ALL());        
    Kokkos::deep_copy(outputSubX2, outputForX2);


}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{
    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", map_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts, summarizedPts);

    // CoeffGradImpl of map
    map_->CoeffGradImpl(summarizedPts, sens, output);
     
}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{
    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", map_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts, summarizedPts);

    // CoeffGradImpl of map
    map_->LogDeterminantCoeffGradImpl(summarizedPts, output);
    
}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{

    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", map_->inputDim, pts.extent(1));

    // Summarize the pts 
    this->SummarizePts(pts, summarizedPts);

    // Create a view for the gradient of map_ wrt s(x_1) and x_2
    Kokkos::View<double**, MemorySpace> outputForSummaryAndX2("outputForSummaryAndX2", map_->inputDim, pts.extent(1));
  
    // GradientImpl of map
    map_->LogDeterminantInputGradImpl(summarizedPts, outputForSummaryAndX2);

    // Split outputForSummaryAndX2 into summary and x2 parts
    Kokkos::View<double**, MemorySpace> outputForSummary = Kokkos::subview(outputForSummaryAndX2, std::make_pair(0,int(summaryFunction_->outputDim)), Kokkos::ALL());
    Kokkos::View<double**, MemorySpace> outputForX2 = Kokkos::subview(outputForSummaryAndX2, std::make_pair(int(summaryFunction_->outputDim),int(summaryFunction_->outputDim+1)), Kokkos::ALL());
    
    // Create a view for the gradient of map_ wrt x_1
    Kokkos::View<double**, MemorySpace> outputForX1("outputForX1", summaryFunction_->inputDim, pts.extent(1));

    // GradientImpl of summary using outputForSummary as the sens. vectors (from chain-rule)
    StridedMatrix<const double, MemorySpace>  x1 = Kokkos::subview(pts, std::make_pair(0,int(summaryFunction_->inputDim)), Kokkos::ALL());
    summaryFunction_->GradientImpl(x1, outputForSummary, outputForX1);

    // Copy results over to the output
    Kokkos::View<double**, MemorySpace> outputSubX1 = Kokkos::subview(output, std::make_pair(0,int(summaryFunction_->inputDim)), Kokkos::ALL());
    Kokkos::deep_copy(outputSubX1, outputForX1);

    Kokkos::View<double**, MemorySpace> outputSubX2 = Kokkos::subview(output, std::make_pair(int(summaryFunction_->inputDim),int(summaryFunction_->inputDim+1)), Kokkos::ALL());        
    Kokkos::deep_copy(outputSubX2, outputForX2);    
}

// Explicit template instantiation
template class mpart::SummarizedMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::SummarizedMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif