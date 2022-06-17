#include <mexplus.h>
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MexArrayConversions.h"
#include "mexplus_eigen.h"
#include <Eigen/Dense>


using namespace mpart;
using namespace mexplus;


// Instance manager for Multi_idxs_tr.
template class mexplus::Session<MultiIndexSet>;

namespace {
MEX_DEFINE(newMutliIndexSetEigen) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const auto multis = input.get<Eigen::MatrixXd>(0);
  output.set(0, Session<MultiIndexSet>::create(new MultiIndexSet(multis.cast<int>())));
}

// Defines MEX API for delete.
MEX_DEFINE(deleteMultiIndexSet) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<MultiIndexSet>::destroy(input.get(0));
}

// Defines MEX API for delete.
MEX_DEFINE(MaxOrders) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.MaxOrders());
}
} // namespace

MEX_DISPATCH // Don't forget to add this if MEX_DEFINE() is used.