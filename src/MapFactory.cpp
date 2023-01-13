#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/SummarizedMap.h"
#include "MParT/AffineFunction.h"
#include "MParT/IdentityMap.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/HermiteFunction.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/LinearizedBasis.h"

using namespace mpart;


template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateComponent(FixedMultiIndexSet<MemorySpace> const& mset,
                                                           MapOptions                                   opts)
{
    return CompFactoryImpl<MemorySpace>::GetFactoryFunction(opts)(mset,opts);
}


template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateSingleEntryMap(unsigned int dim,
                                                                                         unsigned int activeInd,
                                                                                         std::shared_ptr<ConditionalMapBase<MemorySpace>> const &comp)
{

    // Check that active index is not greater than map dimension
    if(dim < activeInd){
        std::stringstream msg;
        msg << "In CreateSingleEntryMap, the active index can't be greater than map dimension. Got dim = " << dim << " and activeInd = " << activeInd << ".";
        throw std::invalid_argument(msg.str());
    }

    // Check that the input dimension of the component matches the activeInd
    if(activeInd != comp->inputDim){
        std::stringstream msg;
        msg << "In CreateSingleEntryMap, the component input dimension must be equal to the active index. Got dim = " << comp->inputDim << " and activeInd = " << activeInd << ".";
        throw std::invalid_argument(msg.str());
    }

    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;
    // Construct map using TriangularMap constructor

    if(activeInd == 1){  // special case if activeInd = 1, map is of form [T_1; Id]

        // Bottom identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> botIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(dim, dim-activeInd);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(2);
        blocks.at(0) = comp;
        blocks.at(1) = botIdMap;

        // make map
        output = std::make_shared<TriangularMap<MemorySpace>>(blocks);


    }
    else if (activeInd == dim){  // special case if activeInd = dim, map is of form [Id; T_d]
        // Top identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> topIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(activeInd-1, activeInd-1);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(2);
        blocks.at(0) = topIdMap;
        blocks.at(1) = comp;

        // make map
        output = std::make_shared<TriangularMap<MemorySpace>>(blocks);
    }
    else{ // general case, map is of form [Id; T_i; Id]

        // Top identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> topIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(activeInd-1, activeInd-1);

        // Bottom identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> botIdMap = std::make_shared<IdentityMap<Kokkos::HostSpace>>(dim, dim-activeInd);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(3);
        blocks.at(0) = topIdMap;
        blocks.at(1) = comp;
        blocks.at(2) = botIdMap;

        // make map
        output = std::make_shared<TriangularMap<MemorySpace>>(blocks);

    }

    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
    return output;

}


template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateTriangular(unsigned int inputDim,
                                                                                     unsigned int outputDim,
                                                                                     unsigned int totalOrder,
                                                                                     MapOptions options)
{

    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> comps(outputDim);

    unsigned int extraInputs = inputDim - outputDim;

    for(unsigned int i=0; i<outputDim; ++i){
        FixedMultiIndexSet<Kokkos::HostSpace> mset(i+extraInputs+1, totalOrder);
        comps.at(i) = CreateComponent<MemorySpace>(mset.ToDevice<MemorySpace>(), options);
    }
    auto output = std::make_shared<TriangularMap<MemorySpace>>(comps);
    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
    return output;
}


template<typename MemorySpace>
std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> mpart::MapFactory::CreateExpansion(unsigned int outputDim,
                                                                                           FixedMultiIndexSet<MemorySpace> const& mset,
                                                                                           MapOptions                                   opts)
{
    std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> output;

    if(opts.basisType==BasisTypes::ProbabilistHermite){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            ProbabilistHermite basis1d(opts.basisNorm);
            output = std::make_shared<MultivariateExpansion<ProbabilistHermite, MemorySpace>>(outputDim, mset, basis1d);
        }else{
            LinearizedBasis<ProbabilistHermite> basis1d(ProbabilistHermite(opts.basisNorm), opts.basisLB, opts.basisUB);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }else if(opts.basisType==BasisTypes::PhysicistHermite){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            PhysicistHermite basis1d(opts.basisNorm);
            output = std::make_shared<MultivariateExpansion<PhysicistHermite, MemorySpace>>(outputDim, mset, basis1d);
        }else{
            LinearizedBasis<PhysicistHermite> basis1d(PhysicistHermite(opts.basisNorm), opts.basisLB, opts.basisUB);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }else if(opts.basisType==BasisTypes::HermiteFunctions){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            HermiteFunction basis1d;
            output = std::make_shared<MultivariateExpansion<HermiteFunction, MemorySpace>>(outputDim, mset, basis1d);
        }else{
            LinearizedBasis<HermiteFunction> basis1d(opts.basisLB, opts.basisUB);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }

    if(output){
        output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
        return output;
    }

    std::stringstream msg;
    msg << "Could not parse options in CreateExpansion.  Unknown 1d basis type.";
    throw std::runtime_error(msg.str());

    return nullptr;
}


template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateComponent<Kokkos::HostSpace>(FixedMultiIndexSet<Kokkos::HostSpace> const&, MapOptions);
template std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> mpart::MapFactory::CreateExpansion<Kokkos::HostSpace>(unsigned int, FixedMultiIndexSet<Kokkos::HostSpace> const&, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateTriangular<Kokkos::HostSpace>(unsigned int, unsigned int, unsigned int, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSingleEntryMap(unsigned int, unsigned int, std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> const&);

#if defined(MPART_ENABLE_GPU)
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateComponent<DeviceSpace>(FixedMultiIndexSet<DeviceSpace> const&, MapOptions);
    template std::shared_ptr<ParameterizedFunctionBase<DeviceSpace>> mpart::MapFactory::CreateExpansion<DeviceSpace>(unsigned int, FixedMultiIndexSet<DeviceSpace> const&, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateTriangular<DeviceSpace>(unsigned int, unsigned int, unsigned int, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSingleEntryMap(unsigned int, unsigned int, std::shared_ptr<ConditionalMapBase<DeviceSpace>> const&);
#endif