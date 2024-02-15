#include <fstream>
#include "MParT/ParameterizedFunctionBase.h"

#include "CommonJuliaUtilities.h"
#include "JlArrayConversions.h"

void mpart::binding::ParameterizedFunctionBaseWrapper(jlcxx::Module &mod) {
    // ParameterizedFunctionBase
    mod.add_type<ParameterizedFunctionBase<Kokkos::HostSpace>>("ParameterizedFunctionBase")
        .method("CoeffMap" , [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb){ return KokkosToJulia(pfb.Coeffs()); })
        .method("SetCoeffs", [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, jlcxx::ArrayRef<double> v){ 
	    Kokkos::View<const double*, Kokkos::HostSpace> ConstCoeffs = JuliaToKokkos(v));
	    pfb.SetCoeffs(ConstCoeffs); 
	})
        .method("numCoeffs", [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb) { return pfb.numCoeffs; })
        .method("inputDim" , [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb) { return pfb.inputDim; })
        .method("outputDim", [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb) { return pfb.outputDim; })
        .method("Evaluate" , [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, jlcxx::ArrayRef<double,2> pts) {
            unsigned int numPts = size(pts,1);
            unsigned int outDim = pfb.outputDim;
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(outDim, numPts);
            for(int j = 0; j < numPts; j++){
                for(int i = 0; i < outDim; i++){
                    output[j*outDim+i] = 0.0;
                }
            }
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
        .method("Gradient", [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, jlcxx::ArrayRef<double,2> pts, jlcxx::ArrayRef<double,2> sens) {
            unsigned int numPts = size(pts,1);
            unsigned int dim = size(pts,0);
            jlcxx::ArrayRef<double,2> output = jlMalloc<double>(dim, numPts);
            for(int j = 0; j < numPts; j++){
                for(int i = 0; i < dim; i++){
                    output[j*dim+i] = 0.0;
                }
            }
            pfb.GradientImpl(JuliaToKokkos(pts), JuliaToKokkos(sens), JuliaToKokkos(output));
            return output;
        })
        .method("Serialize", [](ParameterizedFunctionBase<Kokkos::HostSpace> &pfb, std::string &filename) {
#if defined(MPART_HAS_CEREAL)
            unsigned int inputDim = pfb.inputDim;
            unsigned int outputDim = pfb.outputDim;
            unsigned int numCoeffs = pfb.numCoeffs;
            auto coeffs = pfb.Coeffs();
            std::ofstream os(filename);
            cereal::BinaryOutputArchive oarchive(os);
            oarchive(inputDim,outputDim,numCoeffs);
            oarchive(coeffs);
#else
            std::cerr << "ParameterizedFunctionBase::Serialize: MParT was not compiled with Cereal support. Operation incomplete." << std::endl;
#endif // MPART_HAS_CEREAL
        })
    ;

    mod.method("__DeserializeMap", [](std::string &filename, jlcxx::ArrayRef<int> dims){
#if defined(MPART_HAS_CEREAL)

            std::ifstream is(filename);
            cereal::BinaryInputArchive archive(is);
            unsigned int inputDim, outputDim, numCoeffs;
            archive(inputDim, outputDim, numCoeffs);
            jlcxx::ArrayRef<double> coeffs_jl = jlMalloc<double>(numCoeffs);
            Kokkos::View<double*, Kokkos::HostSpace> coeffs ("Map coeffs", numCoeffs);
            load(archive, coeffs);
            dims[0] = inputDim; dims[1] = outputDim;
            Kokkos::deep_copy(JuliaToKokkos(coeffs_jl), coeffs);
            return coeffs_jl;
#else
            std::cerr << "DeserializeMap: MParT was not compiled with Cereal support. Operation incomplete." << std::endl;
#endif // MPART_HAS_CEREAL
    });
}
