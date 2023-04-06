#include "MParT/ComposedMap.h"
#include "MParT/ConditionalMapBase.h"

#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

namespace jlcxx {
    // Tell CxxWrap.jl the supertype structure for ConditionalMapBase
    template<> struct SuperType<mpart::ComposedMap<Kokkos::HostSpace>> {typedef mpart::ConditionalMapBase<Kokkos::HostSpace> type;};
}

using namespace mpart;

void mpart::binding::ComposedMapWrapper(jlcxx::Module &mod)
{
    mod.add_type<ComposedMap<Kokkos::HostSpace>>("ComposedMap", jlcxx::julia_base_type<ConditionalMapBase<Kokkos::HostSpace>>());
    mod.method("ComposedMap", [](std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> const& maps){
        std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> ret =  std::make_shared<ComposedMap<Kokkos::HostSpace>>(maps);
        return ret;
    })
    ;
}