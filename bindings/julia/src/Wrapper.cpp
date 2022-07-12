#include "CommonJuliaUtilities.h"
#include "MParT/MultiIndices/MultiIndex.h"

#include <iostream>

using namespace mpart::binding;

JLCXX_MODULE define_julia_module(jlcxx::Module& mod)
{
    CommonUtilitiesWrapper(mod);
    MultiIndexWrapper(mod);
    MapOptionsWrapper(mod);
}