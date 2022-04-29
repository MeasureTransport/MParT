#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansion.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

std::shared_ptr<ConditionalMapBase> mpart::CreateComponent(FixedMultiIndexSet const& mset, 
                                                          std::unordered_map<std::string,std::string> options)
{   
    // Extract the polynomial type
    std::string polyType = GetOption(options,"PolyType", "ProbabilistHermite");
    
    // Extract the positive bijector type
    std::string posType = GetOption(options,"PosType", "SoftPlus");

    // Extract the quadrature options
    std::string quadType = GetOption(options, "QuadType", "AdaptiveSimpson");
   
    if(quadType=="AdaptiveSimpson"){
        double relTol = stof(GetOption(options,"QuadRelTol", "1e-6"));
        double absTol = stof(GetOption(options,"QuadAbsTol", "1e-6"));
        unsigned int maxSub = stoi(GetOption(options,"QuadAbsTol", "30"));

        AdaptiveSimpson quad(maxSub, absTol, relTol, QuadError::First);

        if((polyType=="ProbabilistHermite")&&(posType=="SoftPlus")){
            
            MultivariateExpansion<ProbabilistHermite> expansion(mset);
            std::shared_ptr<ConditionalMapBase> output = std::make_shared<MonotoneComponent<decltype(expansion), SoftPlus, AdaptiveSimpson>>(mset, quad);

            output->Coeffs() = Kokkos::View<double*,Kokkos::HostSpace>("Component Coefficients", mset.Size());
            return output;
        }
    }
    
        
    std::stringstream msg;
    msg << "Could not parse options in CreateComponent.  PolyType=\"" << polyType << "\", PosType=\"" << posType << "\", and QuadType=\"" << quadType << "\"";
    throw std::runtime_error(msg.str());

    return nullptr;
} 