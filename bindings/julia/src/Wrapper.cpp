

#include "MParT/MapFactory.h"

#include "CommonJuliaUtilities.h"
#include "CommonUtilities.h"

#include <Kokkos_Core.hpp>
#include <tuple>
#include <iostream>
#include <memory>

#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

using namespace mpart;

JLCXX_MODULE MParT_julia_module(jlcxx::Module& mod)
{
    binding::CommonUtilitiesWrapper(mod);
    binding::MultiIndexWrapper(mod);
    binding::MapOptionsWrapper(mod);
    binding::ParameterizedFunctionBaseWrapper(mod);
    binding::ConditionalMapBaseWrapper(mod);
    binding::TriangularMapWrapper(mod);
    binding::ComposedMapWrapper(mod);
    binding::AffineMapWrapper(mod);
    binding::AffineFunctionWrapper(mod);
    binding::MapFactoryWrapper(mod);
#if defined(MPART_HAS_NLOPT)
    binding::MapObjectiveWrapper(mod);
    binding::TrainMapWrapper(mod);
    binding::AdaptiveTransportMapWrapper(mod);
#endif // MPART_HAS_NLOPT
}