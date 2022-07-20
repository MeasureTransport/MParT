#include "MParT/ComposedMap.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
ComposedMap<MemorySpace>::ComposedMap(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> const& components) : ConditionalMapBase<MemorySpace>(components.front()->inputDim,
                                                                                                                                                         components.back()->output,
                        std::accumulate(components.begin(), components.end(), 0, [](size_t sum, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& comp){ return sum + comp->numCoeffs; })),
                        comps_(components)
{

    // Check the sizes of all the inputs
    for(unsigned int i=0; i<comps_.size()-1; ++i){
        if(comps_.at(i)->outputDim != comps_.at(i+1)->inputDim){
            std::stringstream msg;
            msg << "In ComposedMap constructor, the output dimension (" << comps_.at(i)->outputDim << ") of component " << i << " is not equal to the input dimension (" << comps_.at(i+1)->inputDim << ").";
            throw std::invalid_argument(msg.str());
        }
    }
}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, MemorySpace> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);

    // Now create subviews for each of the components
    unsigned int cumNumCoeffs = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        assert(cumNumCoeffs+comps_.at(i)->numCoeffs <= this->savedCoeffs.size());

        comps_.at(i)->savedCoeffs = Kokkos::subview(this->savedCoeffs,
            std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs));
        cumNumCoeffs += comps_.at(i)->numCoeffs;
    }
}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                    StridedVector<double, MemorySpace>              output)
{


    // LogDet of first component
    comps_.at(0)->LogDeterminantImpl(pts, output);

    //
    if(comps_.size()==1)
        return;

    // intermediate points and variable to hold logdet increments 
    StridedMatrix<const double, MemorySpace> intermediatePts;
    Kokkos::deep_copy(intermediatePts, pts);

    Kokkos::View<double*, MemorySpace> compDetIncrement("Log Determinant", output.extent(0));
    for(unsigned int i=1; i<comps_.size(); ++i){
        
        comps_.at(i)->EvaluateImpl(intermediatePts, intermediatePts);
        comps_.at(i)->LogDeterminantImpl(intermediatePts, compDetIncrement);
        output += compDetIncrement;
    }

}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    Kokkos::deep_copy(output, pts);
    for(unsigned int i=0; i<comps_.size(); ++i){
        
        comps_.at(i)->EvaluateImpl(output, output);

    }

}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                             StridedMatrix<const double, MemorySpace> const& r,
                                             StridedMatrix<double, MemorySpace>              output)
{

    // Evaluate the output for each component
    Kokkos::deep_copy(output, x1);
    
    for(unsigned int i = comps_.size() - 1; i>=0; i--){
        
        comps_.at(i)->InverseImpl(output, r, output);

    }

}


template<typename MemorySpace>
void ComposedMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    StridedMatrix<const double, MemorySpace> subPts;
    StridedMatrix<const double, MemorySpace> subSens; 
    StridedMatrix<double, MemorySpace> subOut;

    int startOutDim = 0;
    int startParamDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){

        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        subSens = Kokkos::subview(sens, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());

        subOut = Kokkos::subview(output, std::make_pair(startParamDim,int(startParamDim+comps_.at(i)->numCoeffs)), Kokkos::ALL());
        comps_.at(i)->CoeffGradImpl(subPts, subSens, subOut);

        startOutDim += comps_.at(i)->outputDim;
        startParamDim += comps_.at(i)->numCoeffs;
    }
}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                             StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    StridedMatrix<const double, MemorySpace> subPts;
    StridedMatrix<double, MemorySpace> subOut;

    int startParamDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){

        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
       
        subOut = Kokkos::subview(output, std::make_pair(startParamDim,int(startParamDim+comps_.at(i)->numCoeffs)), Kokkos::ALL());
        comps_.at(i)->LogDeterminantCoeffGradImpl(subPts, subOut);

        startParamDim += comps_.at(i)->numCoeffs;
    }
}


// I don't know what this does and why it break make step

// // Explicit template instantiation
// template class mpart::ComposedMap<Kokkos::HostSpace>;
// #if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
//     template class mpart::ComposedMap<Kokkos::DefaultExecutionSpace::memory_space>;
// #endif