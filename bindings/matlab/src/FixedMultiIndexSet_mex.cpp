/** Demonstration of mexplus library.
 *
 * In this example, we create MEX APIs for the hypothetical Database class in
 * Matlab.
 *
 */
#include <mexplus.h>
#include <fstream>
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/Utilities/ArrayConversions.h"

using namespace mpart;
using namespace mexplus;

// Instance manager for Multi_idxs_tr.
template class mexplus::Session<FixedMultiIndexSet<Kokkos::HostSpace>>;

namespace {

// Defines MEX API for new.
MEX_DEFINE(FixedMultiIndexSet_newTotalOrder) (int nlhs, mxArray* plhs[],
                                              int nrhs, const mxArray* prhs[]) {
  
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const unsigned int dim = input.get<unsigned int>(0);
  const unsigned int order = input.get<unsigned int>(1);
  output.set(0, Session<FixedMultiIndexSet<Kokkos::HostSpace>>::create(new FixedMultiIndexSet<Kokkos::HostSpace>(dim,order)));
}

// Defines MEX API for new.
MEX_DEFINE(FixedMultiIndexSet_fromMultiIndexSet) (int nlhs, mxArray* plhs[],
                                                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, Session<FixedMultiIndexSet<Kokkos::HostSpace>>::create(new FixedMultiIndexSet<Kokkos::HostSpace>(mset.Fix())));
}

// Defines MEX API for delete.
MEX_DEFINE(FixedMultiIndexSet_delete) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<FixedMultiIndexSet<Kokkos::HostSpace>>::destroy(input.get(0));
}

MEX_DEFINE(FixedMultiIndexSet_MaxDegrees) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet<Kokkos::HostSpace>& mset = Session<FixedMultiIndexSet<Kokkos::HostSpace>>::getConst(input.get(0));
  output.set(0, KokkosToStd(mset.MaxDegrees()));
}

MEX_DEFINE(FixedMultiIndexSet_IndexToMulti) (int nlhs, mxArray* plhs[],
                         int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet<Kokkos::HostSpace>& mset = Session<FixedMultiIndexSet<Kokkos::HostSpace>>::getConst(input.get(0));
  output.set(0, mset.IndexToMulti(input.get<int>(1)));
}

MEX_DEFINE(FixedMultiIndexSet_MultiToIndex) (int nlhs, mxArray* plhs[],
                          int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet<Kokkos::HostSpace>& mset = Session<FixedMultiIndexSet<Kokkos::HostSpace>>::getConst(input.get(0));
  output.set(0, mset.MultiToIndex(input.get<std::vector<unsigned int>>(1)));
}

MEX_DEFINE(FixedMultiIndexSet_Print) (int nlhs, mxArray* plhs[],
                   int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  const FixedMultiIndexSet<Kokkos::HostSpace>& mset = Session<FixedMultiIndexSet<Kokkos::HostSpace>>::getConst(input.get(0));
  mset.Print();
}

MEX_DEFINE(FixedMultiIndexSet_Size) (int nlhs, mxArray* plhs[],
                  int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet<Kokkos::HostSpace>& mset = Session<FixedMultiIndexSet<Kokkos::HostSpace>>::getConst(input.get(0));
  output.set(0, mset.Size());
}

MEX_DEFINE(FixedMultiIndexSet_Length) (int nlhs, mxArray* plhs[],
                int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet<Kokkos::HostSpace>& mset = Session<FixedMultiIndexSet<Kokkos::HostSpace>>::getConst(input.get(0));
  output.set(0, mset.Length());
}

MEX_DEFINE(FixedMultiIndexSet_Serialize) (int nlhs, mxArray* plhs[],
                int nrhs, const mxArray* prhs[]) {
#if defined(MPART_HAS_CEREAL)
  InputArguments input(nrhs, prhs, 2);
  const FixedMultiIndexSet<Kokkos::HostSpace>& mset = Session<FixedMultiIndexSet<Kokkos::HostSpace>>::getConst(input.get(0));
  const std::string& filename = Session<std::string>::getConst(input.get(1));
  std::ofstream os(filename);
  cereal::BinaryOutputArchive oarchive(os);
  oarchive(mset);
#else
  mexErrMsgIdAndTxt("MParT:NoCereal",
                    "MParT was not compiled with Cereal support.");
#endif
}

MEX_DEFINE(FixedMultiIndexSet_Deserialize) (int nlhs, mxArray* plhs[],
                int nrhs, const mxArray* prhs[]) {
#if defined(MPART_HAS_CEREAL)
  InputArguments input(nrhs, prhs, 2);
  OutputArguments output(nlhs, plhs, 1);
  const FixedMultiIndexSet<Kokkos::HostSpace>& mset = Session<FixedMultiIndexSet<Kokkos::HostSpace>>::getConst(input.get(0));
  std::string filename = input.get<std::string>(1);
  std::ifstream is(filename);
  cereal::BinaryInputArchive iarchive(is);
  iarchive(mset);
#else
  mexErrMsgIdAndTxt("MParT:NoCereal",
                    "MParT was not compiled with Cereal support.");
#endif
}

} // namespace
