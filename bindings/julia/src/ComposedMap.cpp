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
    mod.add_type<ComposedMap<Kokkos::HostSpace>>("ComposedMap", jlcxx::julia_base_type<ConditionalMapBase<Kokkos::HostSpace>>())
        .constructor<std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>> const&>()
        // .method("EvaluateUntilK", [](ComposedMap<Kokkos::HostSpace> &map, int k, jlcxx::ArrayRef<double,2>& intPts, jlcxx::ArrayRef<double,2>& output){
        //     return KokkosToJulia(map.EvaluateUntilK(k, JuliaToKokkos(intPts), JuliaToKokkos(output)));
        // })
        ;
}