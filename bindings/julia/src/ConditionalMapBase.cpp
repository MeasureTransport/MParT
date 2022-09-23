#include "MParT/ConditionalMapBase.h"

#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

namespace jlcxx {
    // Tell CxxWrap.jl the supertype structure for ConditionalMapBase
    template<> struct SuperType<mpart::ConditionalMapBase<Kokkos::HostSpace>> {typedef mpart::ParameterizedFunctionBase<Kokkos::HostSpace> type;};
    #if defined(MPART_ENABLE_GPU)
    template<> struct SuperType<mpart::ConditionalMapBase<mpart::DeviceSpace>> {typedef mpart::ParameterizedFunctionBase<mpart::DeviceSpace> type;};
    #endif // MPART_ENABLE_GPU
}

template<typename MemorySpace>
void mpart::binding::ConditionalMapBaseWrapper(jlcxx::Module &mod) {
    // ConditionalMapBase
    std::string tName = "ConditionalMapBase";
    if(!std::is_same<MemorySpace,Kokkos::HostSpace>::value) tName = "d" + tName;

    mod.add_type<ConditionalMapBase<MemorySpace>>(tName, jlcxx::julia_base_type<ParameterizedFunctionBase<MemorySpace>>())
        .method("GetBaseFunction", &ConditionalMapBase<MemorySpace>::GetBaseFunction)
        .method("LogDeterminant", [](ConditionalMapBase<MemorySpace> &map, jlcxx::ArrayRef<double,2> pts){
            unsigned int numPts = size(pts,1);
            jlcxx::ArrayRef<double> output = jlMalloc<double>(numPts);
            StridedMatrix<double, MemorySpace> output_k = JuliaToKokkos<MemorySpace>(output)
            map.LogDeterminantImpl(JuliaToKokkos(pts), output_k);

            return output;
        })
        .method("LogDeterminantCoeffGrad", [](ConditionalMapBase<MemorySpace>& map, jlcxx::ArrayRef<double,2> pts){
            unsigned int numPts = size(pts,1);
            unsigned int numCoeffs = map.numCoeffs;
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(numCoeffs, numPts);
            map.LogDeterminantCoeffGradImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
            return output;
        })
        .method("Inverse", [](ConditionalMapBase<MemorySpace> &map, jlcxx::ArrayRef<double,2> x1, jlcxx::ArrayRef<double,2> r) {
            unsigned int numPts = size(r,1);
            unsigned int outputDim = map.outputDim;
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(outputDim, numPts);
            map.InverseImpl(JuliaToKokkos(x1), JuliaToKokkos(r), JuliaToKokkos(output));
            return output;
        })
        ;
    // jlcxx::stl::apply_stl<ConditionalMapBase<Kokkos::HostSpace>>(mod);
}