#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansion.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

std::shared_ptr<ConditionalMapBase> mpart::CreateComponent(FixedMultiIndexSet<Kokkos::HostSpace> const& mset, 
                                                           MapOptions                                   opts)
{   
    if(opts.quadType==QuadTypes::AdaptiveSimpson){
        
        AdaptiveSimpson<Kokkos::HostSpace> quad(opts.quadMaxSub, 1, nullptr, opts.quadAbsTol, opts.quadRelTol, QuadError::First);

        if(opts.basisType==BasisTypes::ProbabilistHermite){
            
            MultivariateExpansion<ProbabilistHermite> expansion(mset);
            std::shared_ptr<ConditionalMapBase> output;

            switch(opts.posFuncType) {
                case PosFuncTypes::SoftPlus:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad)>>(mset, quad);
                case PosFuncTypes::Exp:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad)>>(mset, quad);
            }
          
            output->Coeffs() = Kokkos::View<double*,Kokkos::HostSpace>("Component Coefficients", mset.Size());
            return output;

        }else if(opts.basisType==BasisTypes::PhysicistHermite){
            
            MultivariateExpansion<PhysicistHermite> expansion(mset);
            std::shared_ptr<ConditionalMapBase> output;

            switch(opts.posFuncType) {
                case PosFuncTypes::SoftPlus:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, decltype(quad)>>(mset, quad);
                case PosFuncTypes::Exp:
                    output = std::make_shared<MonotoneComponent<decltype(expansion), Exp, decltype(quad)>>(mset, quad);
            }
          
            output->Coeffs() = Kokkos::View<double*,Kokkos::HostSpace>("Component Coefficients", mset.Size());
            return output;
        }
    }
    
        
    std::stringstream msg;
    msg << "Could not parse options in CreateComponent.";
    throw std::runtime_error(msg.str());

    return nullptr;
} 