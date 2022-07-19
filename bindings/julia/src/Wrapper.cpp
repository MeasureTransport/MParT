

#include "MParT/MapFactory.h"

#include "CommonJuliaUtilities.h"
#include "CommonUtilities.h"

#include <Kokkos_Core.hpp>
#include <tuple>
#include <iostream>
#include <memory>

#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

using namespace mpart::binding;

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    CommonUtilitiesWrapper(mod);
    MultiIndexWrapper(mod);
    MapOptionsWrapper(mod);
    ParameterizedFunctionBaseWrapper(mod);
    ConditionalMapBaseWrapper(mod);
    TriangularMapWrapper(mod);
    MapFactoryWrapper(mod);
}