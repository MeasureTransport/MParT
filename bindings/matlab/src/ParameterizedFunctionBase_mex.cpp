#include <mexplus.h>
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/ParameterizedFunctionBase.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MexArrayConversions.h"
#include "mexplus_eigen.h"
#include <Eigen/Dense>


using namespace mpart;
using namespace mexplus;
using MemorySpace = Kokkos::HostSpace;


// Instance manager for ParameterizedFunctionBase
template class mexplus::Session<ParameterizedFunctionBase<MemorySpace>>;


namespace {
MEX_DEFINE(ParameterizedFunctionBase_new) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3);
  OutputArguments output(nlhs, plhs, 1);
  unsigned int inDim = input.get<unsigned int>(0);
  unsigned int outDim = input.get<unsigned int>(1);
  unsigned int nCoeffs = input.get<unsigned int>(2);
  output.set(0, Session<ParameterizedFunctionBase<MemorySpace>>::create(new ParameterizedFunctionBase<MemorySpace>(inDim,outDim,nCoeffs)));
}

// Defines MEX API for delete.
MEX_DEFINE(ParameterizedFunctionBase_delete) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<ParameterizedFunctionBase<MemorySpace>>::destroy(input.get(0));
}

} // namespace