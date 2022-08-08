#include "MParT/ComposedMap.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
ComposedMap<MemorySpace>::ComposedMap(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> const& components) : ConditionalMapBase<MemorySpace>(components.front()->inputDim,
                                                                                                                                                         components.back()->outputDim,
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


    // logdet of first component
    comps_.at(0)->LogDeterminantImpl(pts, output);
    if(comps_.size()==1)
        return;

    // intermediate points and variable to hold logdet increments 
    Kokkos::View<double**, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, MemorySpace>  intPts2("intermediate points 2", pts.extent(0), pts.extent(1));
    Kokkos::deep_copy(intPts1, pts);

    Kokkos::View<double*, MemorySpace> compDetIncrement("Log Determinant", output.extent(0));
    for(unsigned int i=1; i<comps_.size(); ++i){
        
        // Compute x_i = T_{i-1}(x_{i-1})
        comps_.at(i-1)->EvaluateImpl(intPts1, intPts2);

        // Compute logdet for T_{i}(x_i)
        comps_.at(i)->LogDeterminantImpl(intPts2, compDetIncrement);

        // Add to logdet of full map
        for(unsigned int j=0; j<output.size(); ++j)
            output(j) += compDetIncrement(j);

        // update current x_{i-1} <-- x_{i}
        Kokkos::deep_copy(intPts1, intPts2);
    }

}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                              StridedMatrix<double, MemorySpace>              output)
{
    // intermediate output
    Kokkos::View<double**, MemorySpace> intOutput("intermediate output", output.extent(0), output.extent(1));
    
    // Copy points to output, then output = map(output) looped over each component
    Kokkos::deep_copy(output, pts);
    for(unsigned int i=0; i<comps_.size(); ++i){
        
        comps_.at(i)->EvaluateImpl(output, intOutput);
        Kokkos::deep_copy(output,intOutput);
    }

}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                             StridedMatrix<const double, MemorySpace> const& r,
                                             StridedMatrix<double, MemorySpace>              output)
{
    // intermediate r
    Kokkos::View<double**, MemorySpace> intR("intermediate r", r.extent(0), r.extent(1));

    // intermediate x1
    Kokkos::View<double**, MemorySpace> intX1("intermediate x1", x1.extent(0), x1.extent(1));

    // Evaluate the output for each component
    Kokkos::deep_copy(intX1, x1);
    Kokkos::deep_copy(intR, r);
    for(int i = comps_.size() - 1; i>=0; --i){
        
        comps_.at(i)->InverseImpl(intX1, intR, output);
        Kokkos::deep_copy(intR, output);

    }

}


template<typename MemorySpace>
void ComposedMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{
        std::stringstream msg;
        msg << "ComposedMap CoeffGradImpl not implemented";
        throw std::invalid_argument(msg.str());
}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                             StridedMatrix<double, MemorySpace>              output)
{
        std::stringstream msg;
        msg << "ComposedMap LogDeterminantCoeffGradImpl not implemented";
        throw std::invalid_argument(msg.str());

}



// Explicit template instantiation
template class mpart::ComposedMap<Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template class mpart::ComposedMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif