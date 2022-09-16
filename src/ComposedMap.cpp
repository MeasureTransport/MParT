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
void ComposedMap<MemorySpace>::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase<MemorySpace>::SetCoeffs(coeffs);

    // Now create subviews for each of the components
    unsigned int cumNumCoeffs = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        assert(cumNumCoeffs+comps_.at(i)->numCoeffs <= this->savedCoeffs.size());

        comps_.at(i)->WrapCoeffs(Kokkos::subview(this->savedCoeffs,
            std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs)));
        cumNumCoeffs += comps_.at(i)->numCoeffs;
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
    for(unsigned int i=0; i<comps_.size(); ++i){
        assert(cumNumCoeffs+comps_.at(i)->numCoeffs <= this->savedCoeffs.size());

        comps_.at(i)->savedCoeffs = Kokkos::subview(this->savedCoeffs,
            std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs));
        cumNumCoeffs += comps_.at(i)->numCoeffs;
    }
}
#endif

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
void ComposedMap<MemorySpace>::GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{

    // g = s^T J_n * J_n-1 *...*J_1
    // for i = n,...,1
    //      x* = T_{i-1} o ... o T_1(x)
    //      s_{i-1}^T = s_{i}^T J_i(x*)
    // output: s0

    Kokkos::View<double**, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, MemorySpace>  intPts2("intermediate points 2", pts.extent(0), pts.extent(1));

    Kokkos::View<double**, MemorySpace>  intSens("intermediate Sens", sens.extent(0), sens.extent(1));
    Kokkos::deep_copy(intSens,sens);    
    for(int i = comps_.size() - 1; i>=0; --i){
        
        // x* = T_{i-1} o ... o T_1(x)

        // reset intPts1 to initial pts
        Kokkos::deep_copy(intPts1, pts);
        for(int j = 0; j<i; j++){
            // x = T_j(x)
            comps_.at(j)->EvaluateImpl(intPts1, intPts2);
            Kokkos::deep_copy(intPts1, intPts2);
        }

        //s_{i-1}^T = s_{i}^T J_i(x*)
        comps_.at(i)->GradientImpl(intPts1, intSens, output);
        Kokkos::deep_copy(intSens, output);

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
void ComposedMap<MemorySpace>::EvaluateUntilK( int k, 
                                        StridedMatrix<const double, MemorySpace> const& pts,
                                        StridedMatrix<double, MemorySpace> intPts, 
                                        StridedMatrix<double, MemorySpace> output){


    Kokkos::deep_copy(intPts, pts);
    Kokkos::deep_copy(output, intPts);
    for(int j = 0; j<k; j++){
        // x = T_j(x)
        comps_.at(j)->EvaluateImpl(intPts, output);
        Kokkos::deep_copy(intPts, output);
    }

}






template<typename MemorySpace>
void ComposedMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                               StridedMatrix<const double, MemorySpace> const& sens,
                                               StridedMatrix<double, MemorySpace>              output)
{

    // T(x) = T_{n} o ... o T_1(x), T_i coeffs w_i
    // s^T \nabla_{w_i} T(x) = s^T J_n * J_{n-1} *...*J_{i-1}* \nabla_w_i T_i 

    Kokkos::View<double**, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, MemorySpace> intPts2("intermediate points 2", pts.extent(0), pts.extent(1));

    Kokkos::View<double**, MemorySpace>  intSens1("intermediate sens 1", sens.extent(0), sens.extent(1));
    Kokkos::View<double**, MemorySpace>  intSens2("intermediate sens 2", sens.extent(0), sens.extent(1));
    Kokkos::deep_copy(intSens1, sens);

    StridedMatrix<double, MemorySpace> subOut;
    int endParamDim = this->numCoeffs;   
    for(int i = comps_.size() - 1; i>=0; --i){
        
        // reset intPts1 to initial pts
        Kokkos::deep_copy(intPts1, pts);

        for(int j = 0; j<i; j++){
            // x = T_j(x)
            comps_.at(j)->EvaluateImpl(intPts1, intPts2);
            Kokkos::deep_copy(intPts1, intPts2);
        }
        // intPts1 = T_{i-1} o ... o T_1(x) 

        // finish g_i = s^T \nabla_{w_i} T(x)
        subOut = Kokkos::subview(output, 
                                 std::make_pair(int(endParamDim-comps_.at(i)->numCoeffs), endParamDim), 
                                 Kokkos::ALL());


        // get subview of output (storing gradient wrt w_i)
        comps_.at(i)->CoeffGradImpl(intPts1, intSens1, subOut);

        // increment sens to for next iteration
        //s_{i-1}^T = s_{i}^T J_i(x*)
        comps_.at(i)->GradientImpl(intPts1, intSens1, intSens2);
        Kokkos::deep_copy(intSens1, intSens2);
        endParamDim -= comps_.at(i)->numCoeffs;

    }

}



template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantCoeffGradImplUpdate( int compInd,
                                        int termInd, 
                                        StridedMatrix<const double, MemorySpace> const& pts, 
                                        StridedMatrix<double, MemorySpace> output){


    // compute first sensitive vector 
    Kokkos::View<double**, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, MemorySpace>  intPts2("intermediate points 2", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, MemorySpace>  sens1("first sens (dim x numPts)", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, MemorySpace>  sens2("second sens (dim x numPts)", pts.extent(0), pts.extent(1));

    this->EvaluateUntilK(termInd, pts, intPts1, intPts2);
    comps_.at(termInd)->LogDeterminantInputGradImpl(intPts1, sens1);

    // loop through computing input gradient terms
    for(int i = termInd - 1; i>compInd; --i){

        // x* = T_{i-1} o ... o T_1(x)
        this->EvaluateUntilK(i, pts, intPts1, intPts2);

        //s_{i-1}^T = s_{i}^T J_i(x*)
        comps_.at(i)->GradientImpl(intPts1, sens1, sens2);
        Kokkos::deep_copy(sens1, sens2);
    }

    // last part, coeff gradient of comp.at(compInd)
    this->EvaluateUntilK(compInd, pts, intPts1, intPts2);
    comps_.at(compInd)->CoeffGradImpl(intPts1, sens1, output);

}

template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantCoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                             StridedMatrix<double, MemorySpace>              output)
{

    unsigned const int numComps = comps_.size();

    Kokkos::View<double**, MemorySpace> intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, MemorySpace> intPts2("intermediate points 2", pts.extent(0), pts.extent(1));
    // set intPts1 to initial pts
    Kokkos::deep_copy(intPts1, pts);

    StridedMatrix<double, MemorySpace> subOut;
    StridedMatrix<double, MemorySpace> subOutUpdate;
    // First set the output to the contribution from LogDeterminantCoeffGradImpl from each component
    int startParamDim = 0;   
    for(int i = 0; i<numComps; ++i){

        // get current subOut for coeffs of comp.at(i)
        subOut = Kokkos::subview(output, 
                                 std::make_pair(startParamDim, int(startParamDim+comps_.at(i)->numCoeffs)), 
                                 Kokkos::ALL());
        
        // get subview of output (storing gradient wrt w_i)
        comps_.at(i)->LogDeterminantCoeffGradImpl(intPts1, subOut);

        //intPts1 = T_i(x_{i-1})
        comps_.at(i)->EvaluateImpl(intPts1, intPts2);
        Kokkos::deep_copy(intPts1, intPts2);

        startParamDim += comps_.at(i)->numCoeffs;
    }

    // Loop through terms in the sum
    for(int k = 0; k<numComps; ++k){

        startParamDim = 0;  
        
        // loop over w_i: coeffs of comps_.at(i): Loop ends at k-1
        for(int i = 0; i<k; ++i){

            // get current subOut for coeffs of comp.at(i)
            subOut = Kokkos::subview(output, 
                                 std::make_pair(startParamDim, int(startParamDim+comps_.at(i)->numCoeffs)), 
                                 Kokkos::ALL());
            
            subOutUpdate = Kokkos::View<double**, MemorySpace>("subOutUpdate", subOut.extent(0), subOut.extent(1));

            LogDeterminantCoeffGradImplUpdate(i, k, pts, subOutUpdate);
            for(int idx1 = 0; idx1<subOut.extent(0); ++idx1){
                for(int idx2 = 0; idx2<subOut.extent(1); ++idx2){
                    subOut(idx1, idx2) += subOutUpdate(idx1,idx2);
                }
            }

            startParamDim += comps_.at(i)->numCoeffs;
        }
    
    }

}



template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                           StridedMatrix<double, MemorySpace>              output)
{

    unsigned const int numComps = comps_.size();

    Kokkos::View<double**, MemorySpace>  intPts1("intermediate points 1", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, MemorySpace> intPts2("intermediate points 2", pts.extent(0), pts.extent(1));
    // set intPts1 to initial pts
    Kokkos::deep_copy(intPts1, pts);

    Kokkos::View<double**, MemorySpace> outUpdate("outUpdate", output.extent(0), output.extent(1));
    Kokkos::View<double**, MemorySpace>  sens("sens (dim x numPts)", pts.extent(0), pts.extent(1));

    // loop through computing input gradient terms
    for(unsigned int k = 0; k<numComps; ++k){

        std::cout << k << std::endl;

        this->EvaluateUntilK(k, pts, intPts1, intPts2);
        comps_.at(k)->LogDeterminantInputGradImpl(intPts1, sens);
        Kokkos::deep_copy(outUpdate, sens);

        // loop through computing input gradient terms
        for(int i = k - 1; i>=0; --i){
            std::cout << i << std::endl;

            // x* = T_{i-1} o ... o T_1(x)

            // reset intPts1 to initial pts
            Kokkos::deep_copy(intPts1, pts);

            for(int j = 0; j<i; ++j){
                // x = T_j(x)
                comps_.at(j)->EvaluateImpl(intPts1, intPts2);
                Kokkos::deep_copy(intPts1, intPts2);
            }
            
            //s_{i-1}^T = s_{i}^T J_i(x*)
            comps_.at(i)->GradientImpl(intPts1, sens, outUpdate);
            Kokkos::deep_copy(sens, outUpdate);
        }

        // Update the output
        for(int idx1 = 0; idx1<output.extent(0); ++idx1){
            for(int idx2 = 0; idx2<output.extent(1); ++idx2){
                //std::cout << outUpdate(idx1, idx2) << std::endl;
                output(idx1, idx2) += outUpdate(idx1, idx2);
            }
        }

    }


}

// Explicit template instantiation
template class mpart::ComposedMap<Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template class mpart::ComposedMap<Kokkos::DefaultExecutionSpace::memory_space>;
#endif