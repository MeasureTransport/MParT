#include <mexplus.h>
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndex.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MexArrayConversions.h"
#include "mexplus_eigen.h"
#include <Eigen/Dense>


using namespace mpart;
using namespace mexplus;


// Instance manager for MultiIndex
template class mexplus::Session<MultiIndex>;


namespace {

MEX_DEFINE(MultiIndex_newDefault) (int nlhs, mxArray* plhs[],
                                         int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const unsigned int lengthIn = input.get<unsigned int>(0);
  const unsigned int val = input.get<unsigned int>(1);
  output.set(0, Session<MultiIndex>::create(new MultiIndex(lengthIn,val)));
}

MEX_DEFINE(MultiIndex_newEigen) (int nlhs, mxArray* plhs[],
                                         int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const auto mult = input.get<Eigen::MatrixXd>(0);
  output.set(0, Session<MultiIndex>::create(new MultiIndex(mult.cast<int>())));
}

// Defines MEX API for delete.
MEX_DEFINE(MultiIndex_delete) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<MultiIndex>::destroy(input.get(0));
}

MEX_DEFINE(MultiIndex_Vector) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  output.set(0, multi.Vector());
}

MEX_DEFINE(MultiIndex_Sum) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  output.set(0, multi.Sum());
}

MEX_DEFINE(MultiIndex_Max) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  output.set(0, multi.Max());
}

MEX_DEFINE(MultiIndex_Set) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 3);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndex *multi = Session<MultiIndex>::get(input.get(0));
  unsigned int ind = input.get<unsigned int>(1);
  unsigned int val = input.get<unsigned int>(2);
  output.set(0, multi->Set(ind,val));
}

MEX_DEFINE(MultiIndex_Get) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  unsigned int ind = input.get<unsigned int>(1);
  output.set(0, multi.Get(ind));
}

MEX_DEFINE(MultiIndex_NumNz) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  output.set(0, multi.NumNz());
}

MEX_DEFINE(MultiIndex_String) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  output.set(0, multi.String());
}

MEX_DEFINE(MultiIndex_Length) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  output.set(0, multi.Length());
}

MEX_DEFINE(MultiIndex_Eq) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  const MultiIndex& multi2 = Session<MultiIndex>::getConst(input.get(1));
  output.set(0, multi==multi2);
}

MEX_DEFINE(MultiIndex_Ne) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  const MultiIndex& multi2 = Session<MultiIndex>::getConst(input.get(1));
  output.set(0, multi!=multi2);
}

MEX_DEFINE(MultiIndex_Lt) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  const MultiIndex& multi2 = Session<MultiIndex>::getConst(input.get(1));
  output.set(0, multi<multi2);
}

MEX_DEFINE(MultiIndex_Gt) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  const MultiIndex& multi2 = Session<MultiIndex>::getConst(input.get(1));
  output.set(0, multi>multi2);
}

MEX_DEFINE(MultiIndex_Ge) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  const MultiIndex& multi2 = Session<MultiIndex>::getConst(input.get(1));
  output.set(0, multi>=multi2);
}

MEX_DEFINE(MultiIndex_Le) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(0));
  const MultiIndex& multi2 = Session<MultiIndex>::getConst(input.get(1));
  output.set(0, multi<=multi2);
}



} // namespace