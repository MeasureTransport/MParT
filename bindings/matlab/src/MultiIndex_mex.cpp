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
  const auto multi = input.get<Eigen::MatrixXd>(0);
  output.set(0, Session<MultiIndex>::create(new MultiIndex(multi.cast<int>())));
}

// Defines MEX API for delete.
MEX_DEFINE(MultiIndex_delete) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<MultiIndex>::destroy(input.get(0));
}

MEX_DEFINE(MultiIndex_String) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndex& mset = Session<MultiIndex>::getConst(input.get(0));
  output.set(0, mset.String());
}


} // namespace