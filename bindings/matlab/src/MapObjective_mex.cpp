#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MapObjective.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"

#include "MexWrapperTypes.h"
#include "MexArrayConversions.h"
#include "mexplus_eigen.h"

using namespace mpart;
using namespace mexplus;
using MemorySpace = Kokkos::HostSpace;

namespace {
    // Defines MEX API for new.
    MEX_DEFINE(GaussianKLObjective_newTrain) (int nlhs, mxArray* plhs[],
                                              int nrhs, const mxArray* prhs[]) {

        InputArguments input(nrhs, prhs, 1);
        OutputArguments output(nlhs, plhs, 1);
        auto train = MexToKokkos2d(prhs[0]);
        auto objective = ObjectiveFactory::CreateGaussianKLObjective(train);
        output.set(0, Session<MapObjectiveMex>::create(new MapObjectiveMex(objective)));
    }

    // Defines MEX API for new.
    MEX_DEFINE(GaussianKLObjective_newTrainTest) (int nlhs, mxArray* plhs[],
                                              int nrhs, const mxArray* prhs[]) {

        InputArguments input(nrhs, prhs, 2);
        OutputArguments output(nlhs, plhs, 1);
        auto train = MexToKokkos2d(prhs[0]);
        auto test = MexToKokkos2d(prhs[1]);
        auto objective = ObjectiveFactory::CreateGaussianKLObjective(train, test);
        output.set(0, Session<MapObjectiveMex>::create(new MapObjectiveMex(objective));
    }

    // Defines MEX API for delete.
    MEX_DEFINE(MapObjective_delete) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
    InputArguments input(nrhs, prhs, 1);
    OutputArguments output(nlhs, plhs, 0);
    Session<MapObjectiveMex>::destroy(input.get(0));
    }

}