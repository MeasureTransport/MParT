/** Demonstration of mexplus library.
 *
 * In this example, we create MEX APIs for the hypothetical Database class in
 * Matlab.
 *
 */
#include <mexplus.h>
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;
using namespace mexplus;

// Instance manager for Multi_idxs_tr.
template class mexplus::Session<FixedMultiIndexSet>;

namespace {
// Defines MEX API for new.
MEX_DEFINE(new) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  output.set(0, Session<FixedMultiIndexSet>::create(new FixedMultiIndexSet(input.get<int>(0), input.get<int>(1))));
}

// Defines MEX API for delete.
MEX_DEFINE(delete) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<FixedMultiIndexSet>::destroy(input.get(0));
}

MEX_DEFINE(MaxDegrees) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
  output.set(0, KokkosToStd(mset.MaxDegrees()));
}

MEX_DEFINE(IndexToMulti) (int nlhs, mxArray* plhs[],
                         int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.IndexToMulti(input.get<int>(1)));
}

MEX_DEFINE(MultiToIndex) (int nlhs, mxArray* plhs[],
                          int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.MultiToIndex(input.get<std::vector<unsigned int>>(1)));
}

MEX_DEFINE(Print) (int nlhs, mxArray* plhs[],
                   int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
  mset.Print();
}

MEX_DEFINE(Size) (int nlhs, mxArray* plhs[],
                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.Size());
}

MEX_DEFINE(dim) (int nlhs, mxArray* plhs[],
                int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.dim);
}

MEX_DEFINE(isCompressed) (int nlhs, mxArray* plhs[],
                          int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.isCompressed);
}

} // namespace

MEX_DISPATCH // Don't forget to add this if MEX_DEFINE() is used.
