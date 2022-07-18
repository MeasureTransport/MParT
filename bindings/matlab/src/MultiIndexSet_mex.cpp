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

MEX_DEFINE(MultiIndexSet_IndexToMulti) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  unsigned int activeIndex = input.get<unsigned int>(1);
  output.set(0, Session<MultiIndex>::create(new MultiIndex(mset.IndexToMulti(activeIndex))));
}

MEX_DEFINE(MultiIndexSet_MultiToIndex) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(1));
  output.set(0,mset.MultiToIndex(multi));
}

MEX_DEFINE(MultiIndexSet_Length) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.Length());
}

MEX_DEFINE(MultiIndexSet_MaxOrders) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.MaxOrders());
}

MEX_DEFINE(MultiIndexSet_at) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  int activeIndex = input.get<int>(1);
  output.set(0, Session<MultiIndex>::create(new MultiIndex(mset.at(activeIndex))));
}

MEX_DEFINE(MultiIndexSet_subsref) (int nlhs, mxArray* plhs[],
                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  int activeIndex = input.get<int>(1);
  output.set(0, Session<MultiIndex>::create(new MultiIndex(mset[activeIndex])));
}

MEX_DEFINE(MultiIndexSet_Size) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.Size());
}

MEX_DEFINE(MultiIndexSet_addMultiIndexSet) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  const MultiIndexSet& msetToAdd = Session<MultiIndexSet>::getConst(input.get(1));
  (*mset)+=msetToAdd;
}

MEX_DEFINE(MultiIndexSet_addMultiIndex) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  const MultiIndex& multiToAdd = Session<MultiIndex>::getConst(input.get(1));
  (*mset)+=multiToAdd;
}

MEX_DEFINE(MultiIndexSet_Union) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  const MultiIndexSet& rhs = Session<MultiIndexSet>::getConst(input.get(1));
  output.set(0, mset->Union(rhs));
}

MEX_DEFINE(MultiIndexSet_Activate) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 0);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(1));
  mset->Activate(multi);
}

MEX_DEFINE(MultiIndexSet_AddActive) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(1));
  output.set(0,mset->AddActive(multi));
}

MEX_DEFINE(MultiIndexSet_Expand) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  unsigned int activeInd = input.get<unsigned int>(1);
  output.set(0, mset->Expand(activeInd));
}

// MEX_DEFINE(MultiIndexSet_ExpandAny) (int nlhs, mxArray* plhs[],
//                     int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 1);
//   OutputArguments output(nlhs, plhs, 0);
//   MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
//   output.set(0, mset->Expand());
// }

MEX_DEFINE(MultiIndexSet_ForciblyExpand) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  const unsigned int activeIndex = input.get<unsigned int>(1);
  output.set(0, mset->ForciblyExpand(activeIndex));
}

MEX_DEFINE(MultiIndexSet_ForciblyActivate) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(1));
  output.set(0,mset->ForciblyActivate(multi));
}

MEX_DEFINE(MultiIndexSet_AdmissibleFowardNeighbors) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  unsigned int activeIndex = input.get<unsigned int>(1);
  std::vector<MultiIndex> vecMultiIndex = mset->AdmissibleForwardNeighbors(activeIndex);
  OutputArguments output(nlhs, plhs, 1);
  std::vector<intptr_t> multi_ids(vecMultiIndex.size());
  for (int i=0; i<vecMultiIndex.size();i++){
    multi_ids[i] =  Session<MultiIndex>::create(new MultiIndex(vecMultiIndex[i]));
  }
  output.set(0,multi_ids);
}

MEX_DEFINE(MultiIndexSet_Frontier) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.Frontier());
}

MEX_DEFINE(MultiIndexSet_Margin) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  std::vector<MultiIndex> vecMultiIndex = mset->Margin();
  std::vector<intptr_t> multi_ids(vecMultiIndex.size());
  for (int i=0; i<vecMultiIndex.size();i++){
    multi_ids[i] =  Session<MultiIndex>::create(new MultiIndex(vecMultiIndex[i]));
  }
  output.set(0,multi_ids);
}

MEX_DEFINE(MultiIndexSet_ReducedMargin) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  MultiIndexSet *mset = Session<MultiIndexSet>::get(input.get(0));
  std::vector<MultiIndex> vecMultiIndex = mset->ReducedMargin();
  std::vector<intptr_t> multi_ids(vecMultiIndex.size());
  for (int i=0; i<vecMultiIndex.size();i++){
    multi_ids[i] =  Session<MultiIndex>::create(new MultiIndex(vecMultiIndex[i]));
  }
  output.set(0,multi_ids);
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

MEX_DEFINE(MultiIndexSet_IsAdmissible) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(1));
  output.set(0, mset.IsAdmissible(multi));
}

MEX_DEFINE(MultiIndexSet_IsExpandable) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  unsigned int activeIndex = input.get<unsigned int>(1);
  output.set(0, mset.IsExpandable(activeIndex));
}

MEX_DEFINE(MultiIndexSet_IsActive) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  const MultiIndex& multi = Session<MultiIndex>::getConst(input.get(1));
  output.set(0, mset.IsActive(multi));
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