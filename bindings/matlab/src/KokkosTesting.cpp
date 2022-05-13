/** Demonstration of mexplus library.
 *
 * In this example, we create MEX APIs for the hypothetical Database class in
 * Matlab.
 *
 */
#include <mexplus.h>
#include "MexArrayConversions.h"

using namespace mpart;
using namespace mexplus;

class TestExp{
public:

    void Evaluate1d(Kokkos::View<const double*, Kokkos::HostSpace> x, Kokkos::View<double*, Kokkos::HostSpace> out) const
    {
        for(unsigned int i=0; i<x.extent(0); ++i){
            out(i) = exp(x(i));
        }
    }
};

// Instance manager for Multi_idxs_tr.
template class mexplus::Session<TestExp>;

namespace {
// Defines MEX API for new.
MEX_DEFINE(new) (int nlhs, mxArray* plhs[],
                 int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 0);
  OutputArguments output(nlhs, plhs, 1);
  output.set(0, Session<TestExp>::create(new TestExp()));
}

// Defines MEX API for delete.
MEX_DEFINE(delete) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Session<TestExp>::destroy(input.get(0));
}

MEX_DEFINE(Evaluate1d) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const TestExp& expObj = Session<TestExp>::getConst(input.get(0));

    // Create a new matlab array to hold the output
    size_t rows = mxGetN(input.get(1));
    mxArray *out = mxCreateDoubleMatrix(rows, 1, mxREAL);

  // Call the evaluate function, which will fill in the matlab array
  expObj.Evaluate1d(MexToKokkos1d(input.get(1)), MexToKokkos1d(out));

  // Return the matlab array
  output.set(0,out);//out);
}

} // namespace

MEX_DISPATCH // Don't forget to add this if MEX_DEFINE() is used.
