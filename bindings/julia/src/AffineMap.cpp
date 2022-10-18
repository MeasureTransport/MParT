#include "MParT/AffineMap.h"
#include "MParT/AffineFunction.h"
#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/ConditionalMapBase.h"

#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

namespace jlcxx {
    // Tell CxxWrap.jl the supertype structure for ConditionalMapBase
    template<> struct SuperType<mpart::AffineMap<Kokkos::HostSpace>> {typedef mpart::ConditionalMapBase<Kokkos::HostSpace> type;};
    template<> struct SuperType<mpart::AffineFunction<Kokkos::HostSpace>>{typedef mpart::ParameterizedFunctionBase<Kokkos::HostSpace> type;};
}

using namespace mpart;

void mpart::binding::AffineMapWrapper(jlcxx::Module &mod)
{
    mod.add_type<AffineMap<Kokkos::HostSpace>>("AffineMap", jlcxx::julia_base_type<ConditionalMapBase<Kokkos::HostSpace>>());
    mod.method("AffineMap", [](jlcxx::ArrayRef<double> b){
        return std::make_shared<AffineMap<Kokkos::HostSpace>>(JuliaToKokkos(b));
    });
    mod.method("AffineMap", [](jlcxx::ArrayRef<double,2> A, jlcxx::ArrayRef<double> b){
        return std::make_shared<AffineMap<Kokkos::HostSpace>>(JuliaToKokkos(A), JuliaToKokkos(b));
    });
    mod.method("AffineMap", [](jlcxx::ArrayRef<double,2> A){
        return std::make_shared<AffineMap<Kokkos::HostSpace>>(JuliaToKokkos(A));
    });
}


void mpart::binding::AffineFunctionWrapper(jlcxx::Module &mod)
{
    mod.add_type<AffineFunction<Kokkos::HostSpace>>("AffineFunction", jlcxx::julia_base_type<ParameterizedFunctionBase<Kokkos::HostSpace>>());
    mod.method("AffineFunction", [](jlcxx::ArrayRef<double> b){
        return std::make_shared<AffineFunction<Kokkos::HostSpace>>(JuliaToKokkos(b));
    });
    mod.method("AffineFunction", [](jlcxx::ArrayRef<double,2> A, jlcxx::ArrayRef<double> b){
        return std::make_shared<AffineFunction<Kokkos::HostSpace>>(JuliaToKokkos(A), JuliaToKokkos(b));
    });
    mod.method("AffineFunction", [](jlcxx::ArrayRef<double,2> A){
        return std::make_shared<AffineFunction<Kokkos::HostSpace>>(JuliaToKokkos(A));
    });
}
