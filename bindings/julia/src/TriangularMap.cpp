#include "MParT/TriangularMap.h"

#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

namespace jlcxx {
    // Tell CxxWrap.jl the supertype structure for TriangularMap
    template<> struct SuperType<mpart::TriangularMap<Kokkos::HostSpace>> {typedef mpart::ConditionalMapBase<Kokkos::HostSpace> type;};
}

void mpart::binding::TriangularMapWrapper(jlcxx::Module &mod) {
    mod.add_type<TriangularMap<Kokkos::HostSpace>>("TriangularMap", jlcxx::julia_base_type<ParameterizedFunctionBase<Kokkos::HostSpace>>())
       .constructor<std::vector<std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>>>>()
       .method("InverseInplace", [](TriangularMap<Kokkos::HostSpace> &map, jlcxx::ArrayRef<double,2> x, jlcxx::ArrayRef<double,2> r){
            map.InverseInplace(JuliaToKokkos(x), JuliaToKokkos(r));
       })
       .method("GetComponent", &TriangularMap<Kokkos::HostSpace>::GetComponent)
    ;
}