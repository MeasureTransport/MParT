#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
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
    if(opts.quadType==QuadTypes::AdaptiveSimpson){

        AdaptiveSimpson<MemorySpace> quad(opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

        if(opts.basisType==BasisTypes::ProbabilistHermite){
            
            if(isinf(opts.basisLB) && isinf(opts.basisUB)){
               
                ProbabilistHermite basis1d(opts.normalizePolys);
                MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;

            }else{

                LinearizedBasis<ProbabilistHermite> basis1d(ProbabilistHermite(opts.normalizePolys), opts.basisLB, opts.basisUB);
                MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;
            }

            
        }else if(opts.basisType==BasisTypes::PhysicistHermite){

            if(isinf(opts.basisLB) && isinf(opts.basisUB)){
                
                PhysicistHermite basis1d(opts.normalizePolys);
                MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;

            }else{

                LinearizedBasis<PhysicistHermite> basis1d(PhysicistHermite(opts.normalizePolys), opts.basisLB, opts.basisUB);
                MultivariateExpansionWorker<decltype(basis1d), MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;
            }

        }else if(opts.basisType==BasisTypes::HermiteFunctions){
            if(isinf(opts.basisLB) && isinf(opts.basisUB)){
                
                MultivariateExpansionWorker<HermiteFunction, MemorySpace> expansion(mset);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                }

                output->Coeffs() = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
                return output;
            }else{

                
                LinearizedBasis<HermiteFunction> basis1d(opts.basisLB,opts.basisUB);
                MultivariateExpansionWorker<decltype(basis1d), MemorySpace> expansion(mset,basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                }

                output->Coeffs() = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
                return output;
            }
        }

    }else if(opts.quadType==QuadTypes::ClenshawCurtis){

        ClenshawCurtisQuadrature<MemorySpace> quad(opts.quadPts, 1);
            
        if(opts.basisType==BasisTypes::ProbabilistHermite){
            
            if(isinf(opts.basisLB) && isinf(opts.basisUB)){
                
                ProbabilistHermite basis1d(opts.normalizePolys);
                MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;

            }else{
                
                LinearizedBasis<ProbabilistHermite> basis1d(ProbabilistHermite(opts.normalizePolys), opts.basisLB, opts.basisUB);

                MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;
            }

            
        }else if(opts.basisType==BasisTypes::PhysicistHermite){

            if(isinf(opts.basisLB) && isinf(opts.basisUB)){
                    
                PhysicistHermite basis1d(opts.normalizePolys);
                MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;
            }else{

                LinearizedBasis<PhysicistHermite> basis1d(PhysicistHermite(opts.normalizePolys), opts.basisLB, opts.basisUB);
                MultivariateExpansionWorker<decltype(basis1d), MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;
            }

        }else if(opts.basisType==BasisTypes::HermiteFunctions){
            if(isinf(opts.basisLB) && isinf(opts.basisUB)){
                    
                MultivariateExpansionWorker<HermiteFunction, MemorySpace> expansion(mset);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                }

                output->Coeffs() = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
                return output;
            }else{
                LinearizedBasis<HermiteFunction> basis1d(opts.basisLB,opts.basisUB);
                MultivariateExpansionWorker<decltype(basis1d), MemorySpace> expansion(mset,basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                }

                output->Coeffs() = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
                return output;
            }
        }

    }else if(opts.quadType==QuadTypes::AdaptiveClenshawCurtis){

        unsigned int level = std::log2(opts.quadPts-2);
        AdaptiveClenshawCurtis<MemorySpace> quad(level, opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First, opts.quadMinSub);

        if(opts.basisType==BasisTypes::ProbabilistHermite){
            
            if(isinf(opts.basisLB) && isinf(opts.basisUB)){

                ProbabilistHermite basis1d(opts.normalizePolys);
                MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;

            }else{
                
                LinearizedBasis<ProbabilistHermite> basis1d(ProbabilistHermite(opts.normalizePolys), opts.basisLB, opts.basisUB);

                MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;
            }

            
        }else if(opts.basisType==BasisTypes::PhysicistHermite){

            if(isinf(opts.basisLB) && isinf(opts.basisUB)){
                    
                PhysicistHermite basis1d(opts.normalizePolys);
                MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;
            }else{

                LinearizedBasis<PhysicistHermite> basis1d(PhysicistHermite(opts.normalizePolys), opts.basisLB, opts.basisUB);
                MultivariateExpansionWorker<decltype(basis1d), MemorySpace> expansion(mset, basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv); break;
                }

                output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size()));
                return output;
            }

        }else if(opts.basisType==BasisTypes::HermiteFunctions){
            if(isinf(opts.basisLB) && isinf(opts.basisUB)){
                    
                MultivariateExpansionWorker<HermiteFunction, MemorySpace> expansion(mset);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                }

                output->Coeffs() = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
                return output;
            }else{
                LinearizedBasis<HermiteFunction> basis1d(opts.basisLB,opts.basisUB);
                MultivariateExpansionWorker<decltype(basis1d), MemorySpace> expansion(mset,basis1d);
                std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

                switch(opts.posFuncType) {
                    case PosFuncTypes::SoftPlus:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                    case PosFuncTypes::Exp:
                        output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad), MemorySpace>>(expansion, quad, opts.contDeriv);
                }

                output->Coeffs() = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
                return output;
            }
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
            ProbabilistHermite basis1d(opts.normalizePolys);
            output = std::make_shared<MultivariateExpansion<ProbabilistHermite, MemorySpace>>(outputDim, mset, basis1d);
        }else{
            LinearizedBasis<ProbabilistHermite> basis1d(ProbabilistHermite(opts.normalizePolys), opts.basisLB, opts.basisUB);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }else if(opts.basisType==BasisTypes::PhysicistHermite){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            PhysicistHermite basis1d(opts.normalizePolys);
            output = std::make_shared<MultivariateExpansion<PhysicistHermite, MemorySpace>>(outputDim, mset, basis1d);
        }else{
            LinearizedBasis<PhysicistHermite> basis1d(PhysicistHermite(opts.normalizePolys), opts.basisLB, opts.basisUB);
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
#if defined(MPART_ENABLE_GPU)
    template std::shared_ptr<ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>> mpart::MapFactory::CreateComponent<Kokkos::DefaultExecutionSpace::memory_space>(FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space> const&, MapOptions);
    template std::shared_ptr<ParameterizedFunctionBase<Kokkos::DefaultExecutionSpace::memory_space>> mpart::MapFactory::CreateExpansion<Kokkos::DefaultExecutionSpace::memory_space>(unsigned int, FixedMultiIndexSet<Kokkos::DefaultExecutionSpace::memory_space> const&, MapOptions);
    template std::shared_ptr<ConditionalMapBase<Kokkos::DefaultExecutionSpace::memory_space>> mpart::MapFactory::CreateTriangular<Kokkos::DefaultExecutionSpace::memory_space>(unsigned int, unsigned int, unsigned int, MapOptions);
#endif