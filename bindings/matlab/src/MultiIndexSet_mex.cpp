#include <mexplus.h>
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

#include "MParT/Utilities/ArrayConversions.h"
#include "MexArrayConversions.h"
#include "mexplus_eigen.h"
#include <Eigen/Dense>


using namespace mpart;
using namespace mexplus;


// Instance manager for MultiIndexSet
// To do: bind functions using MultiIndex objects
template class mexplus::Session<MultiIndexSet>;
template class mexplus::Session<FixedMultiIndexSet<Kokkos::HostSpace>>;


namespace {
MEX_DEFINE(MultiIndexSet_newEigen) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const auto multis = input.get<Eigen::MatrixXd>(0);
  output.set(0, Session<MultiIndexSet>::create(new MultiIndexSet(multis.cast<int>())));
}

MEX_DEFINE(MultiIndexSet_newTotalOrder) (int nlhs, mxArray* plhs[],
                                         int nrhs, const mxArray* prhs[]) {

  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const unsigned int dim = input.get<unsigned int>(0);
  const unsigned int order = input.get<unsigned int>(1);
  output.set(0, Session<MultiIndexSet>::create(new MultiIndexSet(MultiIndexSet::CreateTotalOrder(dim,order))));
}

// Defines MEX API for delete.
MEX_DEFINE(MultiIndexSet_delete) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<MultiIndexSet>::destroy(input.get(0));
}

MEX_DEFINE(MultiIndexSet_MaxOrders) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.MaxOrders());
}

MEX_DEFINE(MultiIndexSet_Size) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.Size());
}

MEX_DEFINE(MultiIndexSet_Expand) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  unsigned int activeInd = input.get<unsigned int>(1);
  output.set(0, mset->Expand(activeInd));
}

MEX_DEFINE(MultiIndexSet_ForciblyExpand) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  const unsigned int activeIndex = input.get<unsigned int>(1);
  output.set(0, mset->ForciblyExpand(activeIndex));
}

MEX_DEFINE(MultiIndexSet_Frontier) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.Frontier());
}

MEX_DEFINE(MultiIndexSet_StrictFrontier) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.StrictFrontier());
}

MEX_DEFINE(MultiIndexSet_BackwardNeighbors) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  unsigned int activeIndex = input.get<unsigned int>(1);
  output.set(0, mset.BackwardNeighbors(activeIndex));
}

MEX_DEFINE(MultiIndexSet_IsExpandable) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  unsigned int activeIndex = input.get<unsigned int>(1);
  output.set(0, mset.IsExpandable(activeIndex));
}

MEX_DEFINE(MultiIndexSet_NumActiveForward) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  unsigned int activeInd = input.get<unsigned int>(1);
  output.set(0, mset.NumActiveForward(activeInd));
}

MEX_DEFINE(MultiIndexSet_NumForward) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  unsigned int activeInd = input.get<unsigned int>(1);
  output.set(0, mset.NumForward(activeInd));
}

MEX_DEFINE(MultiIndexSet_Visualize) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  mset.Visualize();
}


} // namespace