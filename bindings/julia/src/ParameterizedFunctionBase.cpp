#include "MParT/ParameterizedFunctionBase.h"

#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

void mpart::binding::ParameterizedFunctionBaseWrapper(jlcxx::Module &mod) {
    // ParameterizedFunctionBase
    mod.add_type<ParameterizedFunctionBase<Kokkos::HostSpace>>("ParameterizedFunctionBase")
        .method("CoeffMap" , [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb){ return KokkosToJulia(pfb.Coeffs()); })
        .method("SetCoeffs", [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, jlcxx::ArrayRef<double> v){ pfb.SetCoeffs(JuliaToKokkos(v)); })
        .method("numCoeffs", [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb) { return pfb.numCoeffs; })
        .method("inputDim" , [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb) { return pfb.inputDim; })
        .method("outputDim", [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb) { return pfb.outputDim; })
        .method("Evaluate" , [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, jlcxx::ArrayRef<double,2> pts) {
            unsigned int numPts = size(pts,1);
            unsigned int outDim = pfb.outputDim;
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(outDim, numPts);
            pfb.EvaluateImpl(JuliaToKokkos(pts), JuliaToKokkos(output));
            return output;
        })
        .method("CoeffGrad", [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, jlcxx::ArrayRef<double,2> pts, jlcxx::ArrayRef<double,2> sens) {
            unsigned int numPts = size(pts,1);
            unsigned int numCoeffs = pfb.numCoeffs;
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(numCoeffs, numPts);
            pfb.CoeffGradImpl(JuliaToKokkos(pts), JuliaToKokkos(sens), JuliaToKokkos(output));
            return output;
        })
    ;
}