
#include <Kokkos_Core.hpp>
#include <mexplus.h>
#include <MParT/Initialization.h>

using namespace std;
using namespace mexplus;

namespace {

// Defines MEX API for new.
MEX_DEFINE(Kokkos_Initialize) (int nlhs, mxArray* plhs[],
                               int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  mpart::Initialize(Kokkos::InitializationSettings()
    .set_num_threads(input.get<int>(0)));
}

}
