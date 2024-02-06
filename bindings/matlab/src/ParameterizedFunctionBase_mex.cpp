#include <fstream>
#include <mexplus.h>
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MexArrayConversions.h"
#include "MexOptionsConversions.h"
#include "MParT/MapOptions.h"
#include "MParT/MapFactory.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/TriangularMap.h"
#include <Eigen/Dense>


using namespace mpart;
using namespace mpart::binding;
using namespace mexplus;
using MemorySpace = Kokkos::HostSpace;

class ParameterizedFunctionMex {       // The class
public:
  std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> fun_ptr;

  ParameterizedFunctionMex(unsigned int outputDim, FixedMultiIndexSet<MemorySpace> const& mset,
                    MapOptions opts){
    fun_ptr = MapFactory::CreateExpansion<MemorySpace>(outputDim,mset,opts);
  }

  ParameterizedFunctionMex(std::shared_ptr<ConditionalMapBase<MemorySpace>> init_ptr){
    fun_ptr = init_ptr;
  }
}; //end class


// Instance manager for ParameterizedFunctionBase
template class mexplus::Session<ParameterizedFunctionBase<MemorySpace>>;


namespace {
MEX_DEFINE(ParameterizedFunction_newMap) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 15);
  OutputArguments output(nlhs, plhs, 1);
  unsigned int outputDim = input.get<unsigned int>(0);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(1));
  MapOptions opts = MapOptionsFromMatlab(input, 2);

  output.set(0, Session<ParameterizedFunctionMex>::create(new ParameterizedFunctionMex(outputDim,mset.Fix(),opts)));
}

// Defines MEX API for delete.
MEX_DEFINE(ParameterizedFunction_delete) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<ParameterizedFunctionMex>::destroy(input.get(0));
}

MEX_DEFINE(ParameterizedFunction_SetCoeffs) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);
  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));
  auto coeffs = MexToKokkos1d(prhs[1]);
  parFunc.fun_ptr->SetCoeffs(coeffs);
}

MEX_DEFINE(ParameterizedFunction_Coeffs) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));
  auto coeffs = KokkosToVec(parFunc.fun_ptr->Coeffs());
  output.set(0,coeffs);
}

MEX_DEFINE(ParameterizedFunction_CoeffMap) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));
  auto coeffs = parFunc.fun_ptr->CoeffMap();
  output.set(0,coeffs);
}

MEX_DEFINE(ParameterizedFunction_numCoeffs) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));
  auto numcoeffs = parFunc.fun_ptr->numCoeffs;
  output.set(0,numcoeffs);
}

MEX_DEFINE(ParameterizedFunction_outputDim) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));
  unsigned int outDim = parFunc.fun_ptr->outputDim;
  output.set(0, outDim);
}

MEX_DEFINE(ParameterizedFunction_inputDim) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));
  unsigned int inDim = parFunc.fun_ptr->inputDim;
  output.set(0, inDim);
}

MEX_DEFINE(ParameterizedFunction_Evaluate) (int nlhs, mxArray* plhs[],
                                     int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3);
  OutputArguments output(nlhs, plhs, 0);

  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));
  StridedMatrix<const double, Kokkos::HostSpace> pts = MexToKokkos2d(prhs[1]);
  StridedMatrix<double, Kokkos::HostSpace> out = MexToKokkos2d(prhs[2]);
  parFunc.fun_ptr->EvaluateImpl(pts, out);
}

MEX_DEFINE(ParameterizedFunction_CoeffGrad) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 4);
  OutputArguments output(nlhs, plhs, 0);

  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));

  auto pts = MexToKokkos2d(prhs[1]);
  auto sens = MexToKokkos2d(prhs[2]);
  auto out = MexToKokkos2d(prhs[3]);

  parFunc.fun_ptr->CoeffGradImpl(pts,sens,out);
}

MEX_DEFINE(ParameterizedFunction_Gradient) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 4);
  OutputArguments output(nlhs, plhs, 0);

  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));

  auto pts = MexToKokkos2d(prhs[1]);
  auto sens = MexToKokkos2d(prhs[2]);
  auto out = MexToKokkos2d(prhs[3]);

  parFunc.fun_ptr->GradientImpl(pts,sens,out);
}

MEX_DEFINE(ParameterizedFunction_DiagonalCoeffIndices) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);

  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));
  std::vector<unsigned int> indices = parFunc.fun_ptr->DiagonalCoeffIndices();
  output.set(0,indices);
}

MEX_DEFINE(ParameterizedFunction_Serialize) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {

#if defined(MPART_HAS_CEREAL)
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);

  const ParameterizedFunctionMex& parFunc = Session<ParameterizedFunctionMex>::getConst(input.get(0));
  unsigned int inputDim = parFunc.fun_ptr->inputDim;
  unsigned int outputDim = parFunc.fun_ptr->outputDim;
  unsigned int numCoeffs = parFunc.fun_ptr->numCoeffs;
  auto coeffs = parFunc.fun_ptr->Coeffs();
  std::string filename = input.get<std::string>(1);
  std::ofstream os(filename);
  cereal::BinaryOutputArchive oarchive(os);
  oarchive(inputDim,outputDim,numCoeffs);
  oarchive(coeffs);
#else
  mexErrMsgIdAndTxt("MParT:NoCereal",
                    "MParT was not compiled with Cereal support.");
#endif // MPART_HAS_CEREAL
}

MEX_DEFINE(ParameterizedFunction_DeserializeMap) (int nlhs, mxArray* plhs[],
                                      int nrhs, const mxArray* prhs[]) {

#if defined(MPART_HAS_CEREAL)
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 3);

  std::string filename = input.get<std::string>(0);
  std::ifstream is(filename);
  cereal::BinaryInputArchive archive(is);
  unsigned int inputDim, outputDim, numCoeffs;
  archive(inputDim, outputDim, numCoeffs);
  Kokkos::View<double*, Kokkos::HostSpace> coeffs ("Map coeffs", numCoeffs);
  load(archive, coeffs);
  output.set(0,inputDim);
  output.set(1,outputDim);
  output.set(2,CopyKokkosToVec(coeffs));
#else
  mexErrMsgIdAndTxt("MParT:NoCereal",
                    "MParT was not compiled with Cereal support.");
#endif // MPART_HAS_CEREAL
}

MEX_DEFINE(MapOptions_Serialize) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {

#if defined(MPART_HAS_CEREAL)
  InputArguments input(nrhs, prhs, 14);
  OutputArguments output(nlhs, plhs, 1);

  std::string filename = input.get<std::string>(0);
  MapOptions opts = MapOptionsFromMatlab(input, 1);
  std::ofstream os (filename);
  cereal::BinaryOutputArchive oarchive(os);
  oarchive(opts);
#else
  mexErrMsgIdAndTxt("MParT:NoCereal",
                    "MParT was not compiled with Cereal support.");
#endif // MPART_HAS_CEREAL
}


MEX_DEFINE(MapOptions_Deserialize) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {

#if defined(MPART_HAS_CEREAL)
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 12);

  std::string filename = input.get<std::string>(0);
  MapOptions opts;
  std::ifstream is(filename);
  cereal::BinaryInputArchive iarchive(is);
  iarchive(opts);
  MapOptionsToMatlab(opts, output);
#else
  mexErrMsgIdAndTxt("MParT:NoCereal",
                    "MParT was not compiled with Cereal support.");
#endif // MPART_HAS_CEREAL
}

} // namespace