#include <mexplus.h>
#include "MexArrayConversions.h"
#include "mexplus_eigen.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MapObjective.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

using namespace mpart;
using namespace mexplus;
using MemorySpace = Kokkos::HostSpace;

// Instance manager for KL objective.
template class mexplus::Session<KLObjective<Kokkos::HostSpace>>;

namespace {
    // Defines MEX API for new.
    MEX_DEFINE(GaussianKLObjective_newTrain) (int nlhs, mxArray* plhs[],
                                              int nrhs, const mxArray* prhs[]) {
  
        InputArguments input(nrhs, prhs, 1);
        OutputArguments output(nlhs, plhs, 1);
        auto train = MexToKokkos2d(prhs[0]);
        unsigned int dim = train.extent(0);
        std::shared_ptr<mpart::GaussianSamplerDensity<MemorySpace>> density = std::make_shared<mpart::GaussianSamplerDensity<MemorySpace>>(dim);
        output.set(0, Session<KLObjective<MemorySpace>>::create(new KLObjective<MemorySpace>(train, density)));
    }
    // Defines MEX API for new.
    MEX_DEFINE(GaussianKLObjective_newTrainTest) (int nlhs, mxArray* plhs[],
                                              int nrhs, const mxArray* prhs[]) {
  
        InputArguments input(nrhs, prhs, 2);
        OutputArguments output(nlhs, plhs, 1);
        auto train = MexToKokkos2d(prhs[0]);
        auto test = MexToKokkos2d(prhs[1]);
        unsigned int dim = train.extent(0);
        std::shared_ptr<mpart::GaussianSamplerDensity<MemorySpace>> density = std::make_shared<mpart::GaussianSamplerDensity<MemorySpace>>(dim);
        output.set(0, Session<KLObjective<MemorySpace>>::create(new KLObjective<MemorySpace>(train, test, density)));
    }

    // Defines MEX API for delete.
    MEX_DEFINE(GaussianKLObjective_delete) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
    InputArguments input(nrhs, prhs, 1);
    OutputArguments output(nlhs, plhs, 0);
    Session<KLObjective<MemorySpace>>::destroy(input.get(0));
    }

}