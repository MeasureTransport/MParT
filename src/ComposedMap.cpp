#include "MParT/ComposedMap.h"

#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/Utilities/LinearAlgebra.h"

#include <numeric>

using namespace mpart;

template<typename MemorySpace>
ComposedMap<MemorySpace>::Checkpointer::Checkpointer(unsigned int maxSaves, 
                                                     StridedMatrix<const double, MemorySpace> initialPts,
                                                     std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>>& comps) : maxSaves_(maxSaves),
                                                                                                                            maps_(comps)
{
    if(maxSaves_==0){
        throw std::runtime_error("In ComposedMap::Checkpointer: The maximum number of checkpoints must be larger than 0.");
    }

    // Store the initial points as the first checkpoint
    checkpoints_.push_back(Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>("x0", initialPts.extent(0), initialPts.extent(1)));
    Kokkos::deep_copy(checkpoints_.back(), initialPts);

    checkpointLayers_.push_back(0);

    // Allocate the workspace 
    workspace1_ = Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>("xi", initialPts.extent(0), initialPts.extent(1));
    workspace2_ = Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>("xj", initialPts.extent(0), initialPts.extent(1));
}

template<typename MemorySpace>
int ComposedMap<MemorySpace>::Checkpointer::GetNextCheckpoint(unsigned int layerInd) const
{
    // ////
    // Adapted from "revolve" source code distributed with ACM algorithm 799
    // ////

    int currLayer = checkpointLayers_.back();
    int oldLayer = currLayer;
    
    //ds = snaps - *check;
    int availSaves = int(maxSaves_) - checkpoints_.size();
    if(availSaves==0){
        return -1;
    }

    int reps = 0;
    int range = 1;
    
    while(range < layerInd+1 - currLayer) { 
        reps += 1;
        range = range*(reps + availSaves)/reps; 
    }
    
    int bino1, bino2, bino3, bino4, bino5;
    bino1 = range*reps/(availSaves+reps);
    bino2 = (availSaves > 1) ? bino1*availSaves/(availSaves+reps-1) : 1;

    if(availSaves == 1)
        bino3 = 0;
    else
        bino3 = (availSaves > 2) ? bino2*(availSaves-1)/(availSaves+reps-2) : 1;

    bino4 = bino2*(reps-1)/availSaves;
    if(availSaves < 3)
        bino5 = 0;
    else
        bino5 = (availSaves > 3) ? bino3*(availSaves-2)/reps : 1;


    if(layerInd-currLayer <= bino1 + bino3){
        currLayer = currLayer + bino4;
    }else{
        if(layerInd-currLayer >= range - bino5){ 
            currLayer = currLayer + bino1; 
        }else{
            currLayer = layerInd-bino2-bino3;
        }
    }

    if (currLayer == oldLayer) 
        currLayer = oldLayer + 1;  

    return currLayer;
}


template<typename MemorySpace>
Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> ComposedMap<MemorySpace>::Checkpointer::GetLayerInput(unsigned int layerInd)
{   
    // Check if the layer asked for is the last checkpoint
    if(layerInd==checkpointLayers_.back()){
        return checkpoints_.back();

    // If the requested layer is before the last checkpoint, free up the checkpoint
    }else if(layerInd<checkpointLayers_.back()){
        checkpoints_.pop_back();
        checkpointLayers_.pop_back();
        return GetLayerInput(layerInd);

    }else{
        
        
        // Figure out how many more checkpoints we can make ("available saves")
        
        // Figure out the index of the next checkpoint 
        int nextCheckLayer = GetNextCheckpoint(layerInd);
        
        // Compute the input by reevaluating from the last checkpoint
        int i = checkpointLayers_.back();
        maps_.at(i)->EvaluateImpl(checkpoints_.back(), workspace1_);
        ++i;
        for(; i<layerInd; ++i){
            
            if(i==nextCheckLayer){
                
                checkpoints_.push_back(Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>("xi", workspace1_.extent(0), workspace1_.extent(1)));
                Kokkos::deep_copy(checkpoints_.back(),workspace1_);

                checkpointLayers_.push_back(i);

                nextCheckLayer = GetNextCheckpoint(layerInd);
            }

            maps_.at(i)->EvaluateImpl(workspace1_, workspace2_);
            simple_swap(workspace1_,workspace2_);
        }

        return workspace1_;
    }
}


template<typename MemorySpace>
ComposedMap<MemorySpace>::ComposedMap(std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> const& maps,
                                      int maxChecks) : ConditionalMapBase<MemorySpace>(maps.front()->inputDim, 
                                                                                       maps.front()->inputDim,
                                                                                       std::accumulate(maps.begin(), maps.end(), 0, [](size_t sum, std::shared_ptr<ConditionalMapBase<MemorySpace>> const& comp){ return sum + comp->numCoeffs; })),
                                                        maps_(maps),
                                                        maxChecks_((maxChecks<=0) ? maps.size() : maxChecks)
{

    // Check the sizes of all the inputs
    for(unsigned int i=0; i<maps_.size()-1; ++i){
        if(maps_.at(i)->inputDim != maps_.at(i)->outputDim || maps_.at(i)->outputDim != maps_.at(i+1)->inputDim){
            std::stringstream msg;
            msg << "In ComposedMap constructor, each map in the composition must be square. Output dimension (" << maps_.at(i)->outputDim << ") of component " << i << " is not equal to the input dimension (" << maps_.at(i)->inputDim << ").";
            throw std::invalid_argument(msg.str());
        }
    }
    if(maps_.at(maps_.size()-1)->inputDim != maps_.at(maps_.size()-1)->outputDim){
        std::stringstream msg;
        msg << "In ComposedMap constructor, each map in the composition must be square. Output dimension (" << maps_.at(maps_.size()-1)->outputDim << ") of component " << maps_.size()-1 << " is not equal to the input dimension (" << maps_.at(maps_.size()-1)->inputDim << ").";
        throw std::invalid_argument(msg.str());
    }
}

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
    
    maps_.at(0)->EvaluateImpl(pts, intPts1);

    for(int j = 1; j<maps_.size(); j++){
        maps_.at(j)->EvaluateImpl(intPts1, intPts2);
        simple_swap(intPts1, intPts2);
    }

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

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens1("intermediate Sens", sens.extent(0), sens.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens2("intermediate Sens", sens.extent(0), sens.extent(1));
    Kokkos::deep_copy(intSens1,sens);    
    
    Checkpointer checker(maxChecks_, pts, maps_);

    for(int i = maps_.size() - 1; i>=0; --i){
        
        // x* = T_{i-1} o ... o T_1(x)
        auto input = checker.GetLayerInput(i); 

        //s_{i-1}^T = s_{i}^T J_i(x*)
        maps_.at(i)->GradientImpl(input, intSens1, intSens2);
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
void ComposedMap<MemorySpace>::CoeffGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                                             StridedMatrix<const double, MemorySpace> const& sens,
                                             StridedMatrix<double, MemorySpace>              output)
{

    // T(x) = T_{n} o ... o T_1(x), T_i coeffs w_i
    // s^T \nabla_{w_i} T(x) = s^T J_n * J_{n-1} *...*J_{i-1}* \nabla_w_i T_i 
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens1("intermediate sens 1", sens.extent(0), sens.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens2("intermediate sens 2", sens.extent(0), sens.extent(1));
    Kokkos::deep_copy(intSens1, sens);

    Checkpointer checker(maxChecks_, pts, maps_);

    StridedMatrix<double, MemorySpace> subOut;
    int endParamDim = this->numCoeffs;   
    for(int i = maps_.size() - 1; i>=0; --i){

        // intPts1 = T_{i-1} o ... o T_1(x) 
        auto input = checker.GetLayerInput(i);

        // finish g_i = s^T \nabla_{w_i} T(x)
        subOut = Kokkos::subview(output, 
                                 std::make_pair(int(endParamDim-maps_.at(i)->numCoeffs), endParamDim), 
                                 Kokkos::ALL());


        // get subview of output (storing gradient wrt w_i)
        maps_.at(i)->CoeffGradImpl(input, intSens1, subOut);

        // increment sens to for next iteration
        //s_{i-1}^T = s_{i}^T J_i(x*)
        if(i>0){
            maps_.at(i)->GradientImpl(input, intSens1, intSens2);
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

    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens1("intermediate Sens", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens2("intermediate Sens", pts.extent(0), pts.extent(1));
    

    // Get the gradient of the log determinant contribution from the last component
    Checkpointer checker(maxChecks_, pts, maps_);
    auto input = checker.GetLayerInput(maps_.size()-1); 
    
    int endParamDim = this->numCoeffs;
    subOut = Kokkos::subview(output, 
                                 std::make_pair(int(endParamDim-maps_.back()->numCoeffs), endParamDim), 
                                 Kokkos::ALL());
    maps_.back()->LogDeterminantCoeffGradImpl(input, subOut);

    // Get the sensitivity of this log determinant term wrt changes in the input
    maps_.back()->LogDeterminantInputGradImpl(input, intSens1);
    
    endParamDim -= maps_.back()->numCoeffs;  

    for(int i = maps_.size() - 2; i>=0; --i){
        
        // Compute input to this layer
        input = checker.GetLayerInput(i);

        // Gradient for direct contribution of these parameters on the log determinant
        subOut = Kokkos::subview(output, 
                                 std::make_pair(int(endParamDim-maps_.at(i)->numCoeffs), endParamDim), 
                                 Kokkos::ALL());
            
        maps_.at(i)->LogDeterminantCoeffGradImpl(input, subOut);

        // Gradient of later log determinant terms on the coefficients of this layer 
        Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace> subOut2("temp", maps_.at(i)->numCoeffs, pts.extent(1));
        maps_.at(i)->CoeffGradImpl(input, intSens1, subOut2);
        subOut += subOut2;

        if(i>0){
            // Gradient wrt input
            maps_.at(i)->GradientImpl(input, intSens1, intSens2);
            simple_swap<decltype(intSens1)>(intSens1, intSens2);

            // Add sensitivity of log determinant to input
            maps_.at(i)->LogDeterminantInputGradImpl(input, intSens2); 
            
            intSens1 += intSens2;   
        }
        endParamDim -= maps_.at(i)->numCoeffs;   
    }
}



template<typename MemorySpace>
void ComposedMap<MemorySpace>::LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                                           StridedMatrix<double, MemorySpace>              output)
{
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens1("intermediate Sens", pts.extent(0), pts.extent(1));
    Kokkos::View<double**, Kokkos::LayoutLeft, MemorySpace>  intSens2("intermediate Sens", pts.extent(0), pts.extent(1));
    
    // Get the gradient of the log determinant contribution from the last component
    Checkpointer checker(maxChecks_, pts, maps_);
    auto input = checker.GetLayerInput(maps_.size()-1);
    
    maps_.back()->LogDeterminantInputGradImpl(input, intSens1);

    for(int i = maps_.size() - 2; i>=0; --i){
        
        // reset intPts1 to initial pts
        input = checker.GetLayerInput(i);

        //s_{i-1}^T = s_{i}^T J_i(x*)
        maps_.at(i)->GradientImpl(input, intSens1, intSens2);
        simple_swap<decltype(intSens1)>(intSens1, intSens2);

        maps_.at(i)->LogDeterminantInputGradImpl(input, intSens2);  
        intSens1 += intSens2;      
    }

    Kokkos::deep_copy(output, intSens1);
}

// Explicit template instantiation
template class mpart::ComposedMap<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)
    template class mpart::ComposedMap<DeviceSpace>;
#endif