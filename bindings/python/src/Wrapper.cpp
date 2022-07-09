#include "CommonPybindUtilities.h"

using namespace mpart::binding;


PYBIND11_MODULE(pympart, m) {

    CommonUtilitiesWrapper(m);
    MultiIndexWrapper(m);
    MapOptionsWrapper(m);

    ParameterizedFunctionBaseWrapper(m);
    ConditionalMapBaseWrapper(m);
    TriangularMapWrapper(m);
    MapFactoryWrapper(m);
    
}