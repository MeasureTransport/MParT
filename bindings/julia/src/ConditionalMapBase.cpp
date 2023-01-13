#include "MParT/ConditionalMapBase.h"

#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

namespace jlcxx {
    // Tell CxxWrap.jl the supertype structure for ConditionalMapBase
    template<> struct SuperType<mpart::ConditionalMapBase<Kokkos::HostSpace>> {typedef mpart::ParameterizedFunctionBase<Kokkos::HostSpace> type;};
}

void mpart::binding::ConditionalMapBaseWrapper(jlcxx::Module &mod) {
    // ConditionalMapBase
    mod.add_type<ConditionalMapBase<Kokkos::HostSpace>>("ConditionalMapBase", jlcxx::julia_base_type<ParameterizedFunctionBase<Kokkos::HostSpace>>())
        .method("GetBaseFunction", &ConditionalMapBase<Kokkos::HostSpace>::GetBaseFunction)
        .method("LogDeterminant", [](ConditionalMapBase<Kokkos::HostSpace> &map, jlcxx::ArrayRef<double,2> pts){
            unsigned int numPts = size(pts,1);
            jlcxx::ArrayRef<double> output = jlMalloc<double>(numPts);
            map.LogDeterminantImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
            return output;
        })
        .method("LogDeterminantCoeffGrad", [](ConditionalMapBase<Kokkos::HostSpace>& map, jlcxx::ArrayRef<double,2> pts){
            unsigned int numPts = size(pts,1);
            unsigned int numCoeffs = map.numCoeffs;
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(numCoeffs, numPts);
            map.LogDeterminantCoeffGradImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
            return output;
        })
        .method("LogDeterminantInputGrad", [](ConditionalMapBase<Kokkos::HostSpace>& map, jlcxx::ArrayRef<double,2> pts){
            unsigned int numPts = size(pts,1);
            unsigned int numInputs = map.inputDim;
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(numInputs, numPts);
            map.LogDeterminantInputGradImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
            return output;
        })
        .method("Inverse", [](ConditionalMapBase<Kokkos::HostSpace> &map, jlcxx::ArrayRef<double,2> x1, jlcxx::ArrayRef<double,2> r) {
            unsigned int numPts = size(r,1);
            unsigned int outputDim = map.outputDim;
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(outputDim, numPts);
            map.InverseImpl(JuliaToKokkos(x1), JuliaToKokkos(r), JuliaToKokkos(output));
            return output;
        })
        .method("Slice", [](ConditionalMapBase<Kokkos::HostSpace>& map, int begin, int end){ return map.Slice(begin-1, end); })
        ;
    jlcxx::stl::apply_stl<ConditionalMapBase<Kokkos::HostSpace>*>(mod);
}