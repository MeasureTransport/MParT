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

  InputArguments input(nrhs, prhs, 14);
  OutputArguments output(nlhs, plhs, 1);
  unsigned int outputDim = input.get<unsigned int>(0);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(1));
  MapOptions opts = MapOptionsFromMatlab(input.get<std::string>(2),input.get<std::string>(3),
                                         input.get<std::string>(4),input.get<double>(5),
                                         input.get<double>(6),input.get<unsigned int>(7),
                                         input.get<unsigned int>(8),input.get<unsigned int>(9),
                                         input.get<bool>(10),input.get<double>(11),input.get<double>(12),input.get<bool>(13));

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

} // namespace