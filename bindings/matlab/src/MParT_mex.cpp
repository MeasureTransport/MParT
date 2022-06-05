/** Demonstration of mexplus library.
 *
 * In this example, we create MEX APIs for the hypothetical Database class in
 * Matlab.
 *
 */
#include <mexplus.h>
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansion.h"
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
  const auto multis = input.get<Eigen::MatrixXi>(0);
  output.set(0, Session<MultiIndexSet>::create(new MultiIndexSet(multis)));
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
  OutputArguments output(nlhs, plhs, 0);
  const MultiIndexSet& mset = Session<MultiIndexSet>::getConst(input.get(0));
  output.set(0, mset.MaxOrders());
}



// MEX_DEFINE(NumCoeffs) (int nlhs, mxArray* plhs[],
//                         int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 1);
//   OutputArguments output(nlhs, plhs, 1);
//   const MultivariateExpansion<ProbabilistHermite>& expansion = Session<MultivariateExpansion<ProbabilistHermite>>::getConst(input.get(0));
//   output.set(0, expansion.NumCoeffs());
// }

// MEX_DEFINE(CacheSize) (int nlhs, mxArray* plhs[],
//                         int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 1);
//   OutputArguments output(nlhs, plhs, 1);
//   const MultivariateExpansion<ProbabilistHermite>& expansion = Session<MultivariateExpansion<ProbabilistHermite>>::getConst(input.get(0));
//   output.set(0, expansion.CacheSize());
// }

// MEX_DEFINE(FillCache1) (int nlhs, mxArray* plhs[],
//                         int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 3);
//   OutputArguments output(nlhs, plhs, 1);
//   const MultivariateExpansion<ProbabilistHermite>& expansion = Session<MultivariateExpansion<ProbabilistHermite>>::getConst(input.get(0));
//   // Need to bind derivative flag! just switch from string!
//   //DerivativeFlags::DerivativeType flag = None
//   std::vector<double> cache = KokkosToStd(MexToKokkos1d(input.get(1)));
//   expansion.FillCache1(&cache[0],MexToKokkos1d(input.get(2)),DerivativeFlags::None);
//   output.set(0,cache); // Need to output the cache because it is not changed in matlab otherwise
// }

// MEX_DEFINE(Evaluate) (int nlhs, mxArray* plhs[],
//                         int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 3);
//   OutputArguments output(nlhs, plhs, 1);
//   const MultivariateExpansion<ProbabilistHermite>& expansion = Session<MultivariateExpansion<ProbabilistHermite>>::getConst(input.get(0));
//   std::vector<double> cache = KokkosToStd(MexToKokkos1d(input.get(1)));
//   output.set(0, expansion.Evaluate(&cache[0],MexToKokkos1d(input.get(2))));
// }



// // ---------- FixedMultiIndexSet related functions -------------

// MEX_DEFINE(newFixedMultiIndexSet) (int nlhs, mxArray* plhs[],
//                  int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 2);
//   OutputArguments output(nlhs, plhs, 1);
//   output.set(0, Session<FixedMultiIndexSet>::create(new FixedMultiIndexSet(input.get<int>(0), input.get<int>(1))));
// }

// // Defines MEX API for delete.
// MEX_DEFINE(deleteFixedMultiIndexSet) (int nlhs, mxArray* plhs[],
//                     int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 1);
//   OutputArguments output(nlhs, plhs, 0);
//   Session<FixedMultiIndexSet>::destroy(input.get(0));
// }

// MEX_DEFINE(MaxDegrees) (int nlhs, mxArray* plhs[],
//                         int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 1);
//   OutputArguments output(nlhs, plhs, 1);
//   const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
//   output.set(0, KokkosToStd(mset.MaxDegrees()));
// }

// MEX_DEFINE(MaxDegrees2) (int nlhs, mxArray* plhs[],
//                         int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 2);
//   OutputArguments output(nlhs, plhs, 2);
//   const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
//   const FixedMultiIndexSet& mset2 = Session<FixedMultiIndexSet>::getConst(input.get(1));
//   output.set(0, KokkosToStd(mset.MaxDegrees()));
//   output.set(1, KokkosToStd(mset2.MaxDegrees())); //you can call any mset2 method here!
// }

// MEX_DEFINE(IndexToMulti) (int nlhs, mxArray* plhs[],
//                          int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 2);
//   OutputArguments output(nlhs, plhs, 1);
//   const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
//   output.set(0, mset.IndexToMulti(input.get<int>(1)));
// }

// MEX_DEFINE(MultiToIndex) (int nlhs, mxArray* plhs[],
//                           int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 2);
//   OutputArguments output(nlhs, plhs, 1);
//   const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
//   output.set(0, mset.MultiToIndex(input.get<std::vector<unsigned int>>(1)));
// }

// MEX_DEFINE(Print) (int nlhs, mxArray* plhs[],
//                    int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 1);
//   OutputArguments output(nlhs, plhs, 0);
//   const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
//   mset.Print();
// }

// MEX_DEFINE(Size) (int nlhs, mxArray* plhs[],
//                   int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 1);
//   OutputArguments output(nlhs, plhs, 1);
//   const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
//   output.set(0, mset.Size());
// }

// MEX_DEFINE(dim) (int nlhs, mxArray* plhs[],
//                 int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 1);
//   OutputArguments output(nlhs, plhs, 1);
//   const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
//   output.set(0, mset.dim);
// }

// MEX_DEFINE(isCompressed) (int nlhs, mxArray* plhs[],
//                           int nrhs, const mxArray* prhs[]) {
//   InputArguments input(nrhs, prhs, 1);
//   OutputArguments output(nlhs, plhs, 1);
//   const FixedMultiIndexSet& mset = Session<FixedMultiIndexSet>::getConst(input.get(0));
//   output.set(0, mset.isCompressed);
// }

} // namespace

MEX_DISPATCH // Don't forget to add this if MEX_DEFINE() is used.