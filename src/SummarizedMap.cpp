#include "MParT/SummarizedMap.h"

#include "MParT/Utilities/KokkosSpaceMappings.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
SummarizedMap<MemorySpace>::SummarizedMap(std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const& summary, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& component) : ConditionalMapBase<MemorySpace>(summary->inputDim + 1, component->outputDim, component->numCoeffs),
                        sumFunc_(summary), comp_(component)
{

    // Check the dimension of sumFunc_ is compatible with dimension of component
    if(comp_->outputDim != 1){
        std::stringstream msg;
        msg << "SummarizedMap currently supports output dimension = 1 only, but was given " << comp_->outputDim << ".";
        throw std::invalid_argument(msg.str());
    }

    if(comp_->inputDim != sumFunc_->outputDim + 1){
        std::stringstream msg;
        msg << "SummarizedMap: input dimension of map component must be 1 + output dimension of sumFunc_, but was given map->inputDim = " << comp_->inputDim << " and sumFunc_->outputDim + 1 = " << sumFunc_->outputDim + 1 << ".";
        throw std::invalid_argument(msg.str());
    }

}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::SetCoeffs(Kokkos::View<const double*, Kokkos::HostSpace> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);
    comp_->WrapCoeffs(this->savedCoeffs);
}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{

    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::WrapCoeffs(coeffs);
    comp_->WrapCoeffs(coeffs);

}

#if defined(MPART_ENABLE_GPU)
template<typename MemorySpace>
void SummarizedMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{

        // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);
    comp_->WrapCoeffs(coeffs);

}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::WrapCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::WrapCoeffs(coeffs);

    comp_->WrapCoeffs(coeffs);
}
#endif


template<typename MemorySpace>
void SummarizedMap<MemorySpace>::SummarizePts(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{
    // Split pts into the part that is summarized and the part that is not

    StridedMatrix<const double, MemorySpace> ptsToSummarize = Kokkos::subview(pts, std::make_pair(0, int(sumFunc_->inputDim)), Kokkos::ALL());
    StridedMatrix<const double, MemorySpace> ptsAfterSummary = Kokkos::subview(pts, std::make_pair(int(sumFunc_->inputDim), int(sumFunc_->inputDim+1)), Kokkos::ALL());


    // Copy summarized pts
    Kokkos::View<double**, MemorySpace> outputSummary = Kokkos::subview(output, std::make_pair(0,int(sumFunc_->outputDim)), Kokkos::ALL());

    // Evaluate summary function
    sumFunc_->EvaluateImpl(ptsToSummarize, outputSummary);

    // Copy non summarized pts
    Kokkos::View<double**, MemorySpace> outputAfterSummary = Kokkos::subview(output, std::make_pair(int(sumFunc_->outputDim),int(sumFunc_->outputDim+1)), Kokkos::ALL());
    Kokkos::deep_copy(outputAfterSummary, ptsAfterSummary);

}




template<typename MemorySpace>
void SummarizedMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                    StridedVector<double, MemorySpace>              output)
{

    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", comp_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts,summarizedPts);

    // LogDeterminant of map
    comp_->LogDeterminantImpl(summarizedPts, output);

}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{
    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", comp_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts, summarizedPts);

    // Evaluate map
    comp_->EvaluateImpl(summarizedPts, output);
}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                             StridedMatrix<const double, MemorySpace> const& r,
                                             StridedMatrix<double, MemorySpace>              output)
{

    // Create a view to hold pts to summarize
    StridedMatrix<const double, MemorySpace> ptsToSummarize = Kokkos::subview(x1, std::make_pair(0, int(sumFunc_->inputDim)), Kokkos::ALL());

    // // Evaluate summary function
    Kokkos::View<double**, MemorySpace> summary = sumFunc_->Evaluate(ptsToSummarize);

    // Invert map
    comp_->InverseImpl(summary, r, output);
}


template<typename MemorySpace>
void SummarizedMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<const double, MemorySpace> const& sens,
                                              StridedMatrix<double, MemorySpace>              output)
{

    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", comp_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts, summarizedPts);

    // Create a view for the gradient of comp_ wrt s(x_1) and x_2
    Kokkos::View<double**, MemorySpace> outputForSummaryAndX2("outputForSummaryAndX2", comp_->inputDim, pts.extent(1));

    // GradientImpl of map
    comp_->GradientImpl(summarizedPts, sens, outputForSummaryAndX2);

    // Split outputForSummaryAndX2 into summary and x2 parts
    Kokkos::View<double**, MemorySpace> outputForSummary = Kokkos::subview(outputForSummaryAndX2, std::make_pair(0,int(sumFunc_->outputDim)), Kokkos::ALL());
    Kokkos::View<double**, MemorySpace> outputForX2 = Kokkos::subview(outputForSummaryAndX2, std::make_pair(int(sumFunc_->outputDim),int(sumFunc_->outputDim+1)), Kokkos::ALL());

    // GradientImpl of summary using outputForSummary as the sens. vectors (from chain-rule)
    StridedMatrix<const double, MemorySpace>  x1 = Kokkos::subview(pts, std::make_pair(0,int(sumFunc_->inputDim)), Kokkos::ALL());

    // Copy results over to the output
    Kokkos::View<double**, MemorySpace> outputSubX1 = Kokkos::subview(output, std::make_pair(0,int(sumFunc_->inputDim)), Kokkos::ALL());
    sumFunc_->GradientImpl(x1, outputForSummary, outputSubX1);

    Kokkos::View<double**, MemorySpace> outputSubX2 = Kokkos::subview(output, std::make_pair(int(sumFunc_->inputDim),int(sumFunc_->inputDim+1)), Kokkos::ALL());
    Kokkos::deep_copy(outputSubX2, outputForX2);

}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{
    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", comp_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts, summarizedPts);

    // CoeffGradImpl of map
    comp_->CoeffGradImpl(summarizedPts, sens, output);
}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{
    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", comp_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts, summarizedPts);

    // CoeffGradImpl of map
    comp_->LogDeterminantCoeffGradImpl(summarizedPts, output);
}

template<typename MemorySpace>
void SummarizedMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                             StridedMatrix<double, MemorySpace>              output)
{
    // Create a view to hold summarized pts
    Kokkos::View<double**, MemorySpace> summarizedPts("summarizedPts", comp_->inputDim, pts.extent(1));

    // Summarize the pts
    this->SummarizePts(pts, summarizedPts);

    // Create a view for the gradient of comp_ wrt s(x_1) and x_2
    Kokkos::View<double**, MemorySpace> outputForSummaryAndX2("outputForSummaryAndX2", comp_->inputDim, pts.extent(1));

    // GradientImpl of map
    comp_->LogDeterminantInputGradImpl(summarizedPts, outputForSummaryAndX2);

    // Split outputForSummaryAndX2 into summary and x2 parts
    Kokkos::View<double**, MemorySpace> outputForSummary = Kokkos::subview(outputForSummaryAndX2, std::make_pair(0,int(sumFunc_->outputDim)), Kokkos::ALL());
    Kokkos::View<double**, MemorySpace> outputForX2 = Kokkos::subview(outputForSummaryAndX2, std::make_pair(int(sumFunc_->outputDim),int(sumFunc_->outputDim+1)), Kokkos::ALL());

    // Create a view for the gradient of comp_ wrt x_1
    Kokkos::View<double**, MemorySpace> outputForX1("outputForX1", sumFunc_->inputDim, pts.extent(1));

    // GradientImpl of summary using outputForSummary as the sens. vectors (from chain-rule)
    StridedMatrix<const double, MemorySpace>  x1 = Kokkos::subview(pts, std::make_pair(0,int(sumFunc_->inputDim)), Kokkos::ALL());
    sumFunc_->GradientImpl(x1, outputForSummary, outputForX1);

    // Copy results over to the output
    Kokkos::View<double**, MemorySpace> outputSubX1 = Kokkos::subview(output, std::make_pair(0,int(sumFunc_->inputDim)), Kokkos::ALL());
    Kokkos::deep_copy(outputSubX1, outputForX1);

    Kokkos::View<double**, MemorySpace> outputSubX2 = Kokkos::subview(output, std::make_pair(int(sumFunc_->inputDim),int(sumFunc_->inputDim+1)), Kokkos::ALL());
    Kokkos::deep_copy(outputSubX2, outputForX2);
}

// Explicit template instantiation
template class mpart::SummarizedMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::SummarizedMap<DeviceSpace>;
#endif