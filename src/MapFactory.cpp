#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/HermiteFunction.h"
#include "MParT/MultivariateExpansion.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateComponent(FixedMultiIndexSet<MemorySpace> const& mset,
                                                           MapOptions                                   opts)
{
    if(opts.quadType==QuadTypes::AdaptiveSimpson){

        AdaptiveSimpson<MemorySpace> quad(opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First);

        if(opts.basisType==BasisTypes::ProbabilistHermite){

            MultivariateExpansion<ProbabilistHermite,MemorySpace> expansion(mset);
            std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

            switch(opts.posFuncType) {
                case PosFuncTypes::SoftPlus:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(mset, quad);
                case PosFuncTypes::Exp:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(mset, quad);
            }

            output->Coeffs() = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
            return output;

        }else if(opts.basisType==BasisTypes::PhysicistHermite){

            MultivariateExpansion<PhysicistHermite, MemorySpace> expansion(mset);
            std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

            switch(opts.posFuncType) {
                case PosFuncTypes::SoftPlus:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(mset, quad);
                case PosFuncTypes::Exp:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(mset, quad);
            }

            output->Coeffs() = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
            return output;

        }else if(opts.basisType==BasisTypes::HermiteFunctions){

            MultivariateExpansion<HermiteFunction, MemorySpace> expansion(mset);
            std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

            switch(opts.posFuncType) {
                case PosFuncTypes::SoftPlus:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(mset, quad);
                case PosFuncTypes::Exp:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(mset, quad);
            }

            output->Coeffs() = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
            return output;
        }
    }


    std::stringstream msg;
    msg << "Could not parse options in CreateComponent.";
    throw std::runtime_error(msg.str());

    return nullptr;
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
        FixedMultiIndexSet<MemorySpace> mset(i+extraInputs+1, totalOrder);
        comps.at(i) = CreateComponent<MemorySpace>(mset, options);
    }

    return std::make_shared<TriangularMap<MemorySpace>>(comps);
}


template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateComponent<Kokkos::HostSpace>(FixedMultiIndexSet<Kokkos::HostSpace> const&, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateTriangular<Kokkos::HostSpace>(unsigned int, unsigned int, unsigned int, MapOptions);
#if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)
    template std::shared_ptr<ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>> mpart::MapFactory::CreateComponent<Kokkos::DefaultExecutionSpace::memory_space>(FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space> const&, MapOptions);
    template std::shared_ptr<ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>> mpart::MapFactory::CreateTriangular<Kokkos::DefaultExecutionSpace::memory_space>(unsigned int, unsigned int, unsigned int, MapOptions);
#endif