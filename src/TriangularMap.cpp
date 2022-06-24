#include "MParT/TriangularMap.h"

#include <numeric>

using namespace mpart;

TriangularMap::TriangularMap(std::vector<std::shared_ptr<ConditionalMapBase>> const& components) : ConditionalMapBase(components.back()->inputDim, 
                                                                                                                      std::accumulate(components.begin(), components.end(), 0, [](size_t sum, std::shared_ptr<ConditionalMapBase> const& comp){ return sum + comp->outputDim; }),
                                                                                                                      std::accumulate(components.begin(), components.end(), 0, [](size_t sum, std::shared_ptr<ConditionalMapBase> const& comp){ return sum + comp->numCoeffs; })),
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


void TriangularMap::SetCoeffs(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    // First, call the ConditionalMapBase version of this function to copy the view into the savedCoeffs member variable
    ConditionalMapBase::SetCoeffs(coeffs);

    // Now create subviews for each of the components
    unsigned int cumNumCoeffs = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){
        comps_.at(i)->savedCoeffs = Kokkos::subview(savedCoeffs, std::make_pair(cumNumCoeffs, cumNumCoeffs+comps_.at(i)->numCoeffs));
        cumNumCoeffs += comps_.at(i)->numCoeffs;
    }
}


void TriangularMap::LogDeterminantImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
                                       Kokkos::View<double*, Kokkos::HostSpace>             &output)
{
    // Evaluate the log determinant for the first component
    Kokkos::View<const double**, Kokkos::HostSpace> subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(0)->inputDim)), Kokkos::ALL());
    comps_.at(0)->LogDeterminantImpl(subPts, output);

    if(comps_.size()==1)
        return;
    
    // Vector to hold log determinant for a single component
    Kokkos::View<double*, Kokkos::HostSpace> compDet("Log Determinant", output.extent(0));

    for(unsigned int i=1; i<comps_.size(); ++i){
        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        comps_.at(i)->LogDeterminantImpl(subPts, compDet);

        // Add to the output
        for(unsigned int j=0; j<output.size(); ++j)
            output(j) += compDet(j);
    }
}


void TriangularMap::EvaluateImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& pts,
                                 Kokkos::View<double**, Kokkos::HostSpace>            & output)
{
    // Evaluate the output for each component
    Kokkos::View<const double**, Kokkos::HostSpace> subPts;
    Kokkos::View<double**, Kokkos::HostSpace> subOut;

    int startOutDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){

        subPts = Kokkos::subview(pts, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        subOut = Kokkos::subview(output, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());
        comps_.at(i)->EvaluateImpl(subPts, subOut);

        startOutDim += comps_.at(i)->outputDim;
    }
}


void TriangularMap::InverseImpl(Kokkos::View<const double**, Kokkos::HostSpace> const& x1, 
                                Kokkos::View<const double**, Kokkos::HostSpace> const& r,
                                Kokkos::View<double**, Kokkos::HostSpace>            & output)
{
    Kokkos::View<double**, Kokkos::HostSpace> fullOut("Full Output", inputDim, x1.extent(1));
    Kokkos::deep_copy(Kokkos::subview(fullOut, std::make_pair(0,int(x1.extent(0))), Kokkos::ALL()), x1);

    InverseInplace(fullOut, r);

    Kokkos::deep_copy(output, Kokkos::subview(fullOut, std::make_pair(inputDim-outputDim,inputDim), Kokkos::ALL()));
}

void TriangularMap::InverseInplace(Kokkos::View<double**, Kokkos::HostSpace> const& x, 
                                   Kokkos::View<const double**, Kokkos::HostSpace> const& r)
{
    // Evaluate the output for each component
    Kokkos::View<const double**, Kokkos::HostSpace> subR;
    Kokkos::View<const double**, Kokkos::HostSpace> subX;
    Kokkos::View<double**, Kokkos::HostSpace> subOut;
    
    int extraInputs = inputDim - outputDim;

    int startOutDim = 0;
    for(unsigned int i=0; i<comps_.size(); ++i){

        subX = Kokkos::subview(x, std::make_pair(0,int(comps_.at(i)->inputDim)), Kokkos::ALL());
        subR = Kokkos::subview(r, std::make_pair(startOutDim,int(startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());
        subOut = Kokkos::subview(x, std::make_pair(extraInputs + startOutDim,int(extraInputs+startOutDim+comps_.at(i)->outputDim)), Kokkos::ALL());

        comps_.at(i)->InverseImpl(subX, subR, subOut);

        startOutDim += comps_.at(i)->outputDim;
    }
}
