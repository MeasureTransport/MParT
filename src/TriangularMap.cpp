#include "MParT/TriangularMap.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
TriangularMap<MemorySpace>::TriangularMap(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> const& components) : ConditionalMapBase<MemorySpace>(components.back()->inputDim,
                        std::accumulate(components.begin(), components.end(), 0, [](size_t sum, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& comp){ return sum + comp->outputDim; }),
                        std::accumulate(components.begin(), components.end(), 0, [](size_t sum, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& comp){ return sum + comp->numCoeffs; })),
                        comps_(components)
{

    // Check the sizes of all the inputs
    for(unsigned int i=0; i<comps_.size(); ++i){
        if(comps_.at(i)->outputDim > comps_.at(i)->inputDim){
            std::stringstream msg;
            msg << "In TriangularMap constructor, the output dimension (" << comps_.at(i)->outputDim << ") of component " << i << " is greater than the input dimension (" << comps_.at(i)->inputDim << ").";
            throw std::invalid_argument(msg.str());
        }
    }

    for(unsigned int i=1; i<comps_.size(); ++i){
        if(comps_.at(i)->inputDim != (comps_.at(i-1)->inputDim + comps_.at(i-1)->outputDim)){
            std::stringstream msg;
            msg << "In TriangularMap constructor, the input dimension of component " << i << " is " << comps_.at(i)->inputDim;
            msg << ", but is expected to be the sum of the input and output dimensions for component " << i-1;
            msg << ", which is " << comps_.at(i-1)->inputDim << " + " << comps_.at(i-1)->outputDim << " = " << comps_.at(i-1)->inputDim + comps_.at(i-1)->outputDim << ".";
            throw std::invalid_argument(msg.str());
        }
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, MemorySpace> coeffs)
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
void TriangularMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                    Kokkos::View<double*, MemorySpace>              output)
{
    // Evaluate the log determinant for the first component
    Kokkos::View<const double**, MemorySpace> subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(0)->inputDim)), Kokkos::ALL());
    comps_.at(0)->LogDeterminantImpl(subPts, output);

    if(comps_.size()==1)
        return;

    // Vector to hold log determinant for a single component
    Kokkos::View<double*, MemorySpace> compDet("Log Determinant", output.extent(0));

    for(unsigned int i=1; i<comps_.size(); ++i){
        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        comps_.at(i)->LogDeterminantImpl(subPts, compDet);

        // Add to the output
        for(unsigned int j=0; j<output.size(); ++j)
            output(j) += compDet(j);
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    Kokkos::View<const double**, MemorySpace> subPts;
    Kokkos::View<double**, MemorySpace> subOut;

    int startOutDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        subOut = Kokkos::subview(output, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());
        comps_.at(i)->EvaluateImpl(subPts, subOut);

        startOutDim += comps_.at(i)->outputDim;
    }
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                             StridedMatrix<const double, MemorySpace> const& r,
                                             StridedMatrix<double, MemorySpace>              output)
{
    unsigned int ipdim = ConditionalMapBase<MemorySpace>::inputDim;
    unsigned int opdim = ConditionalMapBase<MemorySpace>::outputDim;
    Kokkos::View<double**, MemorySpace> fullOut("Full Output", ipdim, x1.extent(1));
    Kokkos::deep_copy(Kokkos::subview(fullOut, std::make_pair(0,int(x1.extent(0))), Kokkos::ALL()), x1);

    InverseInplace(fullOut, r);

    Kokkos::deep_copy(output, Kokkos::subview(fullOut, std::make_pair(ipdim-opdim,ipdim), Kokkos::ALL()));
}

template<typename MemorySpace>
void TriangularMap<MemorySpace>::InverseInplace(StridedMatrix<double, MemorySpace> x,
                                                StridedMatrix<const double, MemorySpace> const& r)
{
    // Evaluate the output for each component
    Kokkos::View<const double**, MemorySpace> subR;
    Kokkos::View<const double**, MemorySpace> subX;
    Kokkos::View<double**, MemorySpace> subOut;

    unsigned int ipdim = ConditionalMapBase<MemorySpace>::inputDim;
    unsigned int opdim = ConditionalMapBase<MemorySpace>::outputDim;
    unsigned int extraInputs = ipdim - opdim;

    int startOutDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        subX = Kokkos::subview(x, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        subR = Kokkos::subview(r, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());
        subOut = Kokkos::subview(x, std::make_pair(int(extraInputs + startOutDim),int(extraInputs+startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());

        comps_.at(i)->InverseImpl(subX, subR, subOut);

        startOutDim += comps_.at(i)->outputDim;
    }
}


template<typename MemorySpace>
void TriangularMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    Kokkos::View<const double**, MemorySpace> subPts;
    Kokkos::View<const double**, MemorySpace> subSens; 
    Kokkos::View<double**, MemorySpace> subOut;

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
void TriangularMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                             StridedMatrix<double, MemorySpace>              output)
{
    // Evaluate the output for each component
    Kokkos::View<const double**, MemorySpace> subPts;
    Kokkos::View<double**, MemorySpace> subOut;

    int startParamDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){

        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
       
        subOut = Kokkos::subview(output, std::make_pair(startParamDim,int(startParamDim+comps_.at(i)->numCoeffs)), Kokkos::ALL());
        comps_.at(i)->LogDeterminantCoeffGradImpl(subPts, subOut);

        startParamDim += comps_.at(i)->numCoeffs;
    }
}

// Explicit template instantiation
template class mpart::TriangularMap<Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template class mpart::TriangularMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif