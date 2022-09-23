#include "MParT/ComposedMap.h"

#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/Utilities/LinearAlgebra.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
ComposedMap<MemorySpace>::ComposedMap(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> const& maps) : ConditionalMapBase<MemorySpace>(maps.front()->inputDim,
                                                                                                                                                         maps.back()->outputDim,
                        std::accumulate(maps.begin(), maps.end(), 0, [](size_t sum, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& comp){ return sum + comp->numCoeffs; })),
                        maps_(maps)
{

    // Check the sizes of all the inputs
    for(unsigned int i=0; i<maps_.size()-1; ++i){
        if(maps_.at(i)->outputDim != maps_.at(i+1)->inputDim){
            std::stringstream msg;
            msg << "In ComposedMap constructor, the output dimension (" << maps_.at(i)->outputDim << ") of component " << i << " is not equal to the input dimension (" << maps_.at(i+1)->inputDim << ").";
            throw std::invalid_argument(msg.str());
        }
    }
}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);

    // Now create subviews for each of the components
    unsigned int cumNumCoeffs = 0;
    for(unsigned int i=0; i<maps_.size(); ++i){
        assert(cumNumCoeffs+maps_.at(i)->numCoeffs <= this->savedCoeffs.size());

        maps_.at(i)->WrapCoeffs(Kokkos::subview(this->savedCoeffs,
            std::make_pair(cumNumCoeffs, cumNumCoeffs+maps_.at(i)->numCoeffs)));
        cumNumCoeffs += maps_.at(i)->numCoeffs;
    }
}

#if defined(MPART_ENABLE_GPU)
template<typename MemorySpace>
void ComposedMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::DefaultExecutionSpace::memory_space> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);

    // Now create subviews for each of the components
    unsigned int cumNumCoeffs = 0;
    for(unsigned int i=0; i<maps_.size(); ++i){
        assert(cumNumCoeffs+maps_.at(i)->numCoeffs <= this->savedCoeffs.size());

        maps_.at(i)->savedCoeffs = Kokkos::subview(this->savedCoeffs,
            std::make_pair(cumNumCoeffs, cumNumCoeffs+maps_.at(i)->numCoeffs));
        cumNumCoeffs += maps_.at(i)->numCoeffs;
    }
}
#endif

template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                                  StridedVector<double, MemorySpace>              output)
{


    // logdet of first component
    maps_.at(0)->LogDeterminantImpl(pts, output);
    if(maps_.size()==1)
        return;

    // intermediate points and variable to hold logdet increments 
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts2("intermediate points 2", pts.extent(0), pts.extent(1));
    Kokkos::deep_copy(intPts1, pts);

    Kokkos::View<double*, Kokkos::LayoutLeft, MemorySpace> compDetIncrement("Log Determinant", output.extent(0));
    for(unsigned int i=1; i<maps_.size(); ++i){
        
        // Compute x_i = T_{i-1}(x_{i-1})
        maps_.at(i-1)->EvaluateImpl(intPts1, intPts2);

        // Compute logdet for T_{i}(x_i)
        maps_.at(i)->LogDeterminantImpl(intPts2, compDetIncrement);

        // Add to logdet of full map
        output += compDetIncrement;

        // x_{i-1} <-- x_{i} without memory copy
        simple_swap<decltype(intPts1)>(intPts1, intPts2);
    }

}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                                            StridedMatrix<double, MemorySpace>              output)
{
    // intermediate output
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts2("intermediate points 2", pts.extent(0), pts.extent(1));
    
    EvaluateUntilK(maps_.size(), pts, intPts2, intPts1);
    Kokkos::deep_copy(output, intPts1);
}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                            StridedMatrix<const double, MemorySpace> const& sens,
                                            StridedMatrix<double, MemorySpace>              output)
{

    // g = s^T J_n * J_n-1 *...*J_1
    // for i = n,...,1
    //      x* = T_{i-1} o ... o T_1(x)
    //      s_{i-1}^T = s_{i}^T J_i(x*)
    // output: s0

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts2("intermediate points 2", pts.extent(0), pts.extent(1));

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens1("intermediate Sens", sens.extent(0), sens.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens2("intermediate Sens", sens.extent(0), sens.extent(1));
    Kokkos::deep_copy(intSens1,sens);    
    
    for(int i = maps_.size() - 1; i>=0; --i){
        
        // x* = T_{i-1} o ... o T_1(x)
        EvaluateUntilK(i, pts, intPts2, intPts1);

        //s_{i-1}^T = s_{i}^T J_i(x*)
        maps_.at(i)->GradientImpl(intPts1, intSens1, intSens2);
        simple_swap<decltype(intSens1)>(intSens1, intSens2);
    }

    Kokkos::deep_copy(output, intSens1);
}


template<typename MemorySpace>
void ComposedMap<MemorySpace>::InverseImpl(StridedMatrix<const double, MemorySpace> const& x1,
                                           StridedMatrix<const double, MemorySpace> const& r,
                                           StridedMatrix<double, MemorySpace>              output)
{
    // We assume each component of the composed map is square, so the x1 input is not used.
    // We can just loop backward through the layers 

    // intermediate vals
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> intR1("intermediate r1", r.extent(0), r.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> intR2("intermediate r21", r.extent(0), r.extent(1));

    // Evaluate the output for each component
    Kokkos::deep_copy(intR1, r);

    for(int i = maps_.size() - 1; i>=0; --i){
        
        maps_.at(i)->InverseImpl(x1, intR1, intR2); // <- Note the x1 doesn't matter here because the layer is square
        simple_swap<decltype(intR1)>(intR1,intR2);
    }

    Kokkos::deep_copy(output, intR1);
}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::EvaluateUntilK( int k, 
                                               StridedMatrix<const double, MemorySpace> const& pts,
                                               Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>& intPts, 
                                               Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>& output){

    Kokkos::deep_copy(output, pts);
    for(int j = 0; j<k; j++){
        maps_.at(j)->EvaluateImpl(output, intPts);
        simple_swap<Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>>(intPts, output);
    }
}






template<typename MemorySpace>
void ComposedMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                             StridedMatrix<const double, MemorySpace> const& sens,
                                             StridedMatrix<double, MemorySpace>              output)
{

    // T(x) = T_{n} o ... o T_1(x), T_i coeffs w_i
    // s^T \nabla_{w_i} T(x) = s^T J_n * J_{n-1} *...*J_{i-1}* \nabla_w_i T_i 

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts2("intermediate points 2", pts.extent(0), pts.extent(1));

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens1("intermediate sens 1", sens.extent(0), sens.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens2("intermediate sens 2", sens.extent(0), sens.extent(1));
    Kokkos::deep_copy(intSens1, sens);

    StridedMatrix<double, MemorySpace> subOut;
    int endParamDim = this->numCoeffs;   
    for(int i = maps_.size() - 1; i>=0; --i){

        // intPts1 = T_{i-1} o ... o T_1(x) 
        EvaluateUntilK(i, pts, intPts2, intPts1);

        // finish g_i = s^T \nabla_{w_i} T(x)
        subOut = Kokkos::subview(output, 
                                 std::make_pair(int(endParamDim-maps_.at(i)->numCoeffs), endParamDim), 
                                 Kokkos::ALL());


        // get subview of output (storing gradient wrt w_i)
        maps_.at(i)->CoeffGradImpl(intPts1, intSens1, subOut);

        // increment sens to for next iteration
        //s_{i-1}^T = s_{i}^T J_i(x*)
        if(i>0){
            maps_.at(i)->GradientImpl(intPts1, intSens1, intSens2);
            simple_swap<decltype(intSens1)>(intSens1,intSens2);
        }
        endParamDim -= maps_.at(i)->numCoeffs;
    }

}


template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                           StridedMatrix<double, MemorySpace>              output)
{
    StridedMatrix<double, MemorySpace> subOut;

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts2("intermediate points 2", pts.extent(0), pts.extent(1));

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens1("intermediate Sens", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens2("intermediate Sens", pts.extent(0), pts.extent(1));
    

    // Get the gradient of the log determinant contribution from the last component
    EvaluateUntilK(maps_.size()-1, pts, intPts2, intPts1);    
    int endParamDim = this->numCoeffs;
    subOut = Kokkos::subview(output, 
                                 std::make_pair(int(endParamDim-maps_.back()->numCoeffs), endParamDim), 
                                 Kokkos::ALL());
    maps_.back()->LogDeterminantCoeffGradImpl(intPts1, subOut);

    // Get the sensitivity of this log determinant term wrt changes in the input
    maps_.back()->LogDeterminantInputGradImpl(intPts1, intSens1);
    
    endParamDim -= maps_.back()->numCoeffs;  

    for(int i = maps_.size() - 2; i>=0; --i){
        
        // Compute input to this layer
        EvaluateUntilK(i, pts, intPts2, intPts1);

        // Gradient for direct contribution of these parameters on the log determinant
        subOut = Kokkos::subview(output, 
                                 std::make_pair(int(endParamDim-maps_.at(i)->numCoeffs), endParamDim), 
                                 Kokkos::ALL());
                                 
        maps_.at(i)->LogDeterminantCoeffGradImpl(intPts1, subOut);

        // Gradient of later log determinant terms on the coefficients of this layer 
        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> subOut2("temp", maps_.at(i)->numCoeffs, pts.extent(1));
        maps_.at(i)->CoeffGradImpl(intPts1, intSens1, subOut2);
        subOut += subOut2;

        if(i>0){
            // Gradient wrt input
            maps_.at(i)->GradientImpl(intPts1, intSens1, intSens2);
            simple_swap<decltype(intSens1)>(intSens1, intSens2);

            // Add sensitivity of log determinant to input
            maps_.at(i)->LogDeterminantInputGradImpl(intPts1, intSens2);  
            intSens1 += intSens2;   
        }
        endParamDim -= maps_.at(i)->numCoeffs;   
    }
}



template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                           StridedMatrix<double, MemorySpace>              output)
{
    
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intPts2("intermediate points 2", pts.extent(0), pts.extent(1));

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens1("intermediate Sens", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens2("intermediate Sens", pts.extent(0), pts.extent(1));
    
    // Get the gradient of the log determinant contribution from the last component
    EvaluateUntilK(maps_.size()-1, pts, intPts2, intPts1);
    
    maps_.back()->LogDeterminantInputGradImpl(intPts1, intSens1);

    
    for(int i = maps_.size() - 2; i>=0; --i){
        
        // reset intPts1 to initial pts
        Kokkos::deep_copy(intPts1, pts);
        for(int j = 0; j<i; j++){
            // x = T_j(x)
            maps_.at(j)->EvaluateImpl(intPts1, intPts2);
            simple_swap<decltype(intPts1)>(intPts1, intPts2);
        }

        //s_{i-1}^T = s_{i}^T J_i(x*)
        maps_.at(i)->GradientImpl(intPts1, intSens1, intSens2);
        simple_swap<decltype(intSens1)>(intSens1, intSens2);

        maps_.at(i)->LogDeterminantInputGradImpl(intPts1, intSens2);  
        intSens1 += intSens2;      
    }

    Kokkos::deep_copy(output, intSens1);
}

// Explicit template instantiation
template class mpart::ComposedMap<Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template class mpart::ComposedMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif