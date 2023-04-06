#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MapOptions.h"
#include "MParT/MapFactory.h"
#include "MParT/Distributions/GaussianSamplerDensity.h"
#include "MParT/MapObjective.h"
#include "MParT/TrainMap.h"

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

        InputArguments input(nrhs, prhs, 2);
        OutputArguments output(nlhs, plhs, 1);
        StridedMatrix<const double, MemorySpace> train = MexToKokkos2d(prhs[0]);
        unsigned int dim = input.get<unsigned int>(1);
        std::shared_ptr<MapObjective<MemorySpace>> objective = ObjectiveFactory::CreateGaussianKLObjective(train, dim);
        output.set(0, Session<MapObjectiveMex>::create(new MapObjectiveMex(objective)));
    }

    // Defines MEX API for new.
    MEX_DEFINE(GaussianKLObjective_newTrainTest) (int nlhs, mxArray* plhs[],
                                              int nrhs, const mxArray* prhs[]) {

        InputArguments input(nrhs, prhs, 3);
        OutputArguments output(nlhs, plhs, 1);
        StridedMatrix<const double, MemorySpace> train = MexToKokkos2d(prhs[0]);
        StridedMatrix<const double, MemorySpace> test = MexToKokkos2d(prhs[1]);
        unsigned int dim = input.get<unsigned int>(2);
        std::shared_ptr<MapObjective<MemorySpace>> objective = ObjectiveFactory::CreateGaussianKLObjective(train, test, dim);
        output.set(0, Session<MapObjectiveMex>::create(new MapObjectiveMex(objective)));
    }

    // Defines MEX API for delete.
    MEX_DEFINE(MapObjective_delete) (int nlhs, mxArray* plhs[],
                        int nrhs, const mxArray* prhs[]) {
        InputArguments input(nrhs, prhs, 1);
        OutputArguments output(nlhs, plhs, 0);
        Session<MapObjectiveMex>::destroy(input.get(0));
    }

}