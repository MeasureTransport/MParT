#include "MParT/IdentityMap.h"

#include <numeric>

using namespace mpart;

IdentityMap::IdentityMap(unsigned int dim) : ConditionalMapBase(dim,dim,0)
{


}


// void TriangularMap::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
// {
//     // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
//     ConditionalMapBase::SetCoeffs(coeffs);

//     // Now create subviews for each of the components
//     unsigned int cumNumCoeffs = 0;
//     for(unsigned int i=0; i<comps_.size(); ++i){
//         comps_.at(i)->savedCoeffs = Kokkos::subview(savedCoeffs, std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs));
//         cumNumCoeffs += comps_.at(i)->numCoeffs;
//     }
// }


void IdentityMap::LogDeterminantImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
                                       Kokkos::View<double*, Kokkos::HostSpace>             &output)
{


    // Add to the output
    for(unsigned int j=0; j<output.size(); ++j)
        output(j) = 0;
}


void IdentityMap::EvaluateImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
                                 Kokkos::View<double**, Kokkos::HostSpace>            & output)
{
    // Evaluate the output


}


// void TriangularMap::InverseImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& x1, 
//                                 Kokkos::View<const double**, Kokkos::HostSpace> const& r,
//                                 Kokkos::View<double**, Kokkos::HostSpace>            & output)
// {
//     Kokkos::View<double**, Kokkos::HostSpace> fullOut("Full Output", inputDim, x1.extent(1));
//     Kokkos::deep_copy(Kokkos::subview(fullOut, std::make_pair(0,int(x1.extent(0))), Kokkos::ALL()), x1);

//     InverseInplace(fullOut, r);

//     Kokkos::deep_copy(output, Kokkos::subview(fullOut, std::make_pair(inputDim-outputDim,inputDim), Kokkos::ALL()));
// }

// void TriangularMap::InverseInplace(Kokkos::View<double**, Kokkos::HostSpace> const& x, 
//                                    Kokkos::View<const double**, Kokkos::HostSpace> const& r)
// {
//     // Evaluate the output for each component
//     Kokkos::View<const double**, Kokkos::HostSpace> subR;
//     Kokkos::View<const double**, Kokkos::HostSpace> subX;
//     Kokkos::View<double**, Kokkos::HostSpace> subOut;
    
//     int extraInputs = inputDim - outputDim;

//     int startOutDim = 0;
//     for(unsigned int i=0; i<comps_.size(); ++i){

//         subX = Kokkos::subview(x, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
//         subR = Kokkos::subview(r, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());
//         subOut = Kokkos::subview(x, std::make_pair(extraInputs + startOutDim,int(extraInputs+startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());

//         comps_.at(i)->InverseImpl(subX, subR, subOut);

//         startOutDim += comps_.at(i)->outputDim;
//     }
// }
