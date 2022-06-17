/** Demonstration of mexplus library.
 *
 * In this example, we create MEX APIs for the hypothetical Database class in
 * Matlab.
 *
 */
#include <mexplus.h>
#include "MParT/Utilities/ArrayConversions.h"
#include "MexArrayConversions.h"
#include "mexplus_eigen.h"
#include "MParT/MapOptions.h"


#include <Eigen/Dense>


using namespace mpart;
using namespace mexplus;

class TestExp{
public:

    MapOption_mex() const
    {
      opts.basisType    = BasisTypes::ProbabilistHermite;
      opts.posFuncType = PosFuncTypes::SoftPlus;
      opts.quadType    = QuadTypes::AdaptiveSimpson;
      opts.quadAbsTol  = 1e-6;
      opts.quadRelTol  = 1e-6;
    }
};


// Instance manager for Multi_idxs_tr.
template class mexplus::Session<BasisTypes>;

namespace {
MEX_DEFINE(new) (int nlhs, mxArray* plhs[],
                    int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 1);
  const auto type = input.get<std::string>(0);
  MapOptions opts;

  output.set(0, opts);
}

} // namespace

MEX_DISPATCH // Don't forget to add this if MEX_DEFINE() is used.