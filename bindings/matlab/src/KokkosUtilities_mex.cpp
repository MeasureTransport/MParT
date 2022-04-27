
#include <Kokkos_Core.hpp>
#include <mexplus.h>

using namespace std;
using namespace mexplus;

namespace {

// Defines MEX API for new.
MEX_DEFINE(Kokkos_Initialize) (int nlhs, mxArray* plhs[],
                               int nrhs, const mxArray* prhs[]) {
  InputArguments input(nrhs, prhs, 1);
  OutputArguments output(nlhs, plhs, 0);
  Kokkos::InitArguments args;
  args.num_threads = input.get<int>(0);
  Kokkos::initialize(args);
}

MEX_DEFINE(Kokkos_Finalize) (int nlhs, mxArray* plhs[],
                               int nrhs, const mxArray* prhs[]) {
  Kokkos::finalize();
}

}

MEX_DISPATCH // Don't forget to add this if MEX_DEFINE() is used.
